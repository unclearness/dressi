package org.dressi.examples

import android.content.Context
import android.content.res.AssetManager
import java.io.File

/**
 * Copies the APK's assets under data/ into filesDir/data on first launch
 * (the native loaders use fopen, so real files are required). Re-extracts
 * when the app version changes.
 */
object AssetInstaller {
    fun install(context: Context): File {
        val dataDir = File(context.filesDir, "data")
        val marker =
            File(context.filesDir, ".assets_v${BuildConfig.VERSION_CODE}")
        if (marker.exists()) {
            return dataDir
        }
        dataDir.deleteRecursively()
        copyAssetPath(context.assets, "data", dataDir)
        context.filesDir.listFiles { f -> f.name.startsWith(".assets_v") }
            ?.forEach { it.delete() }
        marker.createNewFile()
        return dataDir
    }

    private fun copyAssetPath(assets: AssetManager, path: String, dst: File) {
        val children = assets.list(path) ?: return
        if (children.isEmpty()) {
            dst.parentFile?.mkdirs()
            assets.open(path).use { input ->
                dst.outputStream().use { output -> input.copyTo(output) }
            }
        } else {
            for (child in children) {
                copyAssetPath(assets, "$path/$child", File(dst, child))
            }
        }
    }
}
