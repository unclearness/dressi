package org.dressi.examples

import android.view.Surface

object NativeBridge {
    init {
        System.loadLibrary("dressi_examples")
    }

    interface Listener {
        fun onLog(line: String)
        fun onStreamsChanged(titles: Array<String>)
        fun onFinished(name: String, exitCode: Int)
    }

    /** Entries are "name|g" where g=1 means geometryShader is required. */
    external fun listExamples(): Array<String>

    /** JSON: deviceName, apiVersion, geometryShader, limits... */
    external fun queryDeviceCaps(): String

    /** False if another example is still running (one at a time). */
    external fun startExample(name: String, args: Array<String>): Boolean

    /** Cooperative: the example exits at its next iteration boundary. */
    external fun requestStop()

    external fun isRunning(): Boolean

    external fun setSurface(surface: Surface?)

    external fun selectStream(index: Int)

    external fun setListener(listener: Listener?)
}
