plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.dressi.examples"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "org.dressi.examples"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake { arguments += listOf("-DANDROID_STL=c++_static") }
        }
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }
    buildFeatures { buildConfig = true }
    buildTypes {
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

// Bundle the curated dataset subset (repo data/ is populated by a desktop
// CMake configure; the 512x256 EXR comes from tools/downsample_exr).
val syncDressiAssets by tasks.registering(Copy::class) {
    val dataDir = file("../../data")
    doFirst {
        if (!File(dataDir, "bunny/bunny.obj").exists()) {
            logger.warn(
                "WARNING: ../../data is not populated. Run a desktop CMake " +
                "configure once (downloads the datasets), and " +
                "tools/downsample_exr for the small env map. The app " +
                "builds, but examples needing assets stay disabled.")
        }
    }
    from(dataDir) {
        include("bunny/bunny.obj", "bunny/bunny.mtl",
                "bunny/bunny-atlas.jpg")
        include("Avocado/glTF/**")
        include("DamagedHelmet/glTF/**")
        include("suburban_garden_512.exr")
    }
    into(file("src/main/assets/data"))
}
tasks.named("preBuild") { dependsOn(syncDressiAssets) }
