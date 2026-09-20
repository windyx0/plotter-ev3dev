package pth.windyx.plott3r.panel

import android.annotation.SuppressLint
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.LinearLayout
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : AppCompatActivity() {
    private lateinit var webView: WebView
    private var filePathCallback: ValueCallback<Array<Uri>>? = null

    private val fileChooserLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            val data = result.data?.data
            if (data != null) {
                filePathCallback?.onReceiveValue(arrayOf(data))
            } else {
                filePathCallback?.onReceiveValue(null)
            }
        } else {
            filePathCallback?.onReceiveValue(null)
        }
        filePathCallback = null
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        supportActionBar?.hide()
        
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        
        webView = WebView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT
            )
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.cacheMode = WebSettings.LOAD_NO_CACHE
            settings.loadWithOverviewMode = true
            settings.useWideViewPort = true
            settings.allowFileAccess = true
            settings.allowContentAccess = true
            settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            
            webViewClient = WebViewClient()
            webChromeClient = object : WebChromeClient() {
                override fun onShowFileChooser(
                    webView: WebView?,
                    filePathCallback: ValueCallback<Array<Uri>>?,
                    fileChooserParams: FileChooserParams?
                ): Boolean {
                    this@MainActivity.filePathCallback?.onReceiveValue(null)
                    this@MainActivity.filePathCallback = filePathCallback
                    
                    val isImage = fileChooserParams?.acceptTypes?.any { it.contains("image") } == true
                    val intent = if (isImage) {
                        Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                    } else {
                        fileChooserParams?.createIntent() ?: Intent(Intent.ACTION_GET_CONTENT).apply {
                            addCategory(Intent.CATEGORY_OPENABLE)
                            type = "*/*"
                        }
                    }
                    fileChooserLauncher.launch(intent)
                    return true
                }
            }
        }
        
        layout.addView(webView)
        setContentView(layout)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        Thread {
            try {
                val py = Python.getInstance()
                py.getModule("server_wrapper").callAttr("start_server")
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }.start()

        // Show loading splash screen
        val loadingHtml = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {
                        background-color: #041226;
                        color: #e8f1ff;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        font-family: sans-serif;
                    }
                    .spinner {
                        width: 50px;
                        height: 50px;
                        border: 4px solid rgba(47, 126, 255, 0.2);
                        border-top-color: #2f7eff;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin-bottom: 20px;
                    }
                    @keyframes spin { 100% { transform: rotate(360deg); } }
                    h2 { margin: 0; font-weight: 600; letter-spacing: 1px; }
                    p { color: #9db4d6; font-size: 0.9em; margin-top: 8px; }
                </style>
            </head>
            <body>
                <div class="spinner"></div>
                <h2>Plott3r</h2>
                <p>Запуск нейронного движка...</p>
            </body>
            </html>
        """.trimIndent()
        webView.loadDataWithBaseURL(null, loadingHtml, "text/html", "UTF-8", null)

        // Polling thread for Flask server
        Thread {
            var serverUp = false
            for (i in 1..20) {
                try {
                    val url = java.net.URL("http://127.0.0.1:5000/")
                    val connection = url.openConnection() as java.net.HttpURLConnection
                    connection.connectTimeout = 500
                    connection.readTimeout = 500
                    connection.requestMethod = "GET"
                    connection.connect()
                    if (connection.responseCode == 200) {
                        serverUp = true
                        break
                    }
                } catch (e: Exception) { }
                Thread.sleep(500)
            }
            
            runOnUiThread {
                if (serverUp) {
                    webView.loadUrl("http://127.0.0.1:5000")
                } else {
                    val errorHtml = loadingHtml.replace("Запуск нейронного движка...", "Ошибка запуска сервера. Перезапустите приложение.")
                                               .replace("spinner", "")
                    webView.loadDataWithBaseURL(null, errorHtml, "text/html", "UTF-8", null)
                }
            }
        }.start()
    }
}
