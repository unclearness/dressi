// JNI surface for org.dressi.examples.NativeBridge: example registry,
// device caps, one-at-a-time example runner (background std::thread with
// cooperative cancellation), Surface lifecycle, and listener callbacks.

#include <android/native_window_jni.h>
#include <jni.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "android_host.h"
#include "example_host.h"

namespace dressi_android {
std::string QueryDeviceCapsJson();  // device_caps.cpp
}

namespace {

JavaVM* g_vm = nullptr;
jobject g_listener = nullptr;  // global ref
std::mutex g_listener_mutex;

std::atomic<bool> g_cancel{false};
std::atomic<bool> g_running{false};
std::thread g_thread;

// Env for the current thread, attaching if needed (returns whether a
// matching Detach is required).
JNIEnv* GetEnv(bool* needs_detach) {
    *needs_detach = false;
    if (!g_vm) {
        return nullptr;
    }
    JNIEnv* env = nullptr;
    if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) ==
        JNI_OK) {
        return env;
    }
    if (g_vm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
        *needs_detach = true;
        return env;
    }
    return nullptr;
}

// Calls listener.<method>(args...) if a listener is set. The jobject ref
// is cloned under the mutex so setListener can't race the call.
template <typename Fn>
void WithListener(Fn&& fn) {
    jobject listener = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_listener_mutex);
        listener = g_listener;
    }
    if (!listener) {
        return;
    }
    bool detach = false;
    JNIEnv* env = GetEnv(&detach);
    if (!env) {
        return;
    }
    fn(env, listener);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
    }
    if (detach) {
        g_vm->DetachCurrentThread();
    }
}

jstring ToJString(JNIEnv* env, const std::string& s) {
    return env->NewStringUTF(s.c_str());
}

std::string FromJString(JNIEnv* env, jstring s) {
    if (!s) {
        return {};
    }
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    env->ReleaseStringUTFChars(s, c);
    return out;
}

}  // namespace

namespace dressi_android {

void NotifyLog(const std::string& line) {
    WithListener([&](JNIEnv* env, jobject l) {
        jclass cls = env->GetObjectClass(l);
        jmethodID mid = env->GetMethodID(cls, "onLog",
                                         "(Ljava/lang/String;)V");
        jstring js = ToJString(env, line);
        env->CallVoidMethod(l, mid, js);
        env->DeleteLocalRef(js);
        env->DeleteLocalRef(cls);
    });
}

void NotifyStreamsChanged(const std::vector<std::string>& titles) {
    WithListener([&](JNIEnv* env, jobject l) {
        jclass str_cls = env->FindClass("java/lang/String");
        jobjectArray arr = env->NewObjectArray(jsize(titles.size()), str_cls,
                                               nullptr);
        for (size_t i = 0; i < titles.size(); i++) {
            jstring js = ToJString(env, titles[i]);
            env->SetObjectArrayElement(arr, jsize(i), js);
            env->DeleteLocalRef(js);
        }
        jclass cls = env->GetObjectClass(l);
        jmethodID mid = env->GetMethodID(cls, "onStreamsChanged",
                                         "([Ljava/lang/String;)V");
        env->CallVoidMethod(l, mid, arr);
        env->DeleteLocalRef(arr);
        env->DeleteLocalRef(str_cls);
        env->DeleteLocalRef(cls);
    });
}

void NotifyFinished(const std::string& name, int exit_code) {
    WithListener([&](JNIEnv* env, jobject l) {
        jclass cls = env->GetObjectClass(l);
        jmethodID mid = env->GetMethodID(cls, "onFinished",
                                         "(Ljava/lang/String;I)V");
        jstring js = ToJString(env, name);
        env->CallVoidMethod(l, mid, js, jint(exit_code));
        env->DeleteLocalRef(js);
        env->DeleteLocalRef(cls);
    });
}

}  // namespace dressi_android

using dressi_android::AndroidHost;
using dressi_android::SurfaceState;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
    g_vm = vm;
    dressi_android::InstallLogSinks();
    return JNI_VERSION_1_6;
}

JNIEXPORT jobjectArray JNICALL
Java_org_dressi_examples_NativeBridge_listExamples(JNIEnv* env,
                                                        jclass) {
    const auto& registry = dressi_examples::GetExampleRegistry();
    jclass str_cls = env->FindClass("java/lang/String");
    jobjectArray arr = env->NewObjectArray(jsize(registry.size()), str_cls,
                                           nullptr);
    for (size_t i = 0; i < registry.size(); i++) {
        // "name|g" where g=1 marks the geometryShader requirement
        const std::string entry =
                std::string(registry[i].name) + "|" +
                (registry[i].needs_geometry_shader ? "1" : "0");
        jstring js = env->NewStringUTF(entry.c_str());
        env->SetObjectArrayElement(arr, jsize(i), js);
        env->DeleteLocalRef(js);
    }
    return arr;
}

JNIEXPORT jstring JNICALL
Java_org_dressi_examples_NativeBridge_queryDeviceCaps(JNIEnv* env,
                                                           jclass) {
    return env->NewStringUTF(dressi_android::QueryDeviceCapsJson().c_str());
}

JNIEXPORT jboolean JNICALL
Java_org_dressi_examples_NativeBridge_startExample(JNIEnv* env, jclass,
                                                        jstring jname,
                                                        jobjectArray jargs) {
    if (g_running.exchange(true)) {
        return JNI_FALSE;  // one run at a time
    }
    if (g_thread.joinable()) {
        g_thread.join();  // reap the previous (finished) runner
    }
    const std::string name = FromJString(env, jname);
    std::vector<std::string> args;
    const jsize n = jargs ? env->GetArrayLength(jargs) : 0;
    for (jsize i = 0; i < n; i++) {
        auto js = static_cast<jstring>(env->GetObjectArrayElement(jargs, i));
        args.push_back(FromJString(env, js));
        env->DeleteLocalRef(js);
    }
    dressi_examples::RunFn fn = nullptr;
    for (const auto& info : dressi_examples::GetExampleRegistry()) {
        if (name == info.name) {
            fn = info.run;
        }
    }
    if (!fn) {
        g_running = false;
        return JNI_FALSE;
    }
    g_cancel = false;
    SurfaceState::instance().clearStreams();
    g_thread = std::thread([name, args = std::move(args), fn]() {
        JNIEnv* env = nullptr;
        g_vm->AttachCurrentThread(&env, nullptr);
        AndroidHost host(g_cancel);
        int code = 2;
        try {
            code = fn(args, host);  // examples catch their own exceptions
        } catch (...) {
        }
        dressi_android::NotifyFinished(name, code);
        g_running = false;
        g_vm->DetachCurrentThread();
    });
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_org_dressi_examples_NativeBridge_requestStop(JNIEnv*, jclass) {
    g_cancel = true;
}

JNIEXPORT jboolean JNICALL
Java_org_dressi_examples_NativeBridge_isRunning(JNIEnv*, jclass) {
    return g_running.load() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_org_dressi_examples_NativeBridge_setSurface(JNIEnv* env, jclass,
                                                      jobject jsurface) {
    ANativeWindow* window =
            jsurface ? ANativeWindow_fromSurface(env, jsurface) : nullptr;
    SurfaceState::instance().setSurface(window);  // takes ownership
}

JNIEXPORT void JNICALL
Java_org_dressi_examples_NativeBridge_selectStream(JNIEnv*, jclass,
                                                        jint index) {
    SurfaceState::instance().selectStream(int(index));
}

JNIEXPORT void JNICALL
Java_org_dressi_examples_NativeBridge_setListener(JNIEnv* env, jclass,
                                                       jobject listener) {
    std::lock_guard<std::mutex> lock(g_listener_mutex);
    if (g_listener) {
        env->DeleteGlobalRef(g_listener);
        g_listener = nullptr;
    }
    if (listener) {
        g_listener = env->NewGlobalRef(listener);
    }
}

}  // extern "C"
