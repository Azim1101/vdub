package com.azim.vdub

import android.app.Application
import com.azim.vdub.core.VdubPaths
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class VdubApp : Application() {
    override fun onCreate() {
        super.onCreate()
        runCatching { VdubPaths.ensureRoots() }
    }
}
