package com.azim.vdub.core

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import com.azim.vdub.MainActivity
import com.azim.vdub.R

/**
 * Keeps long work alive while the screen is off.
 *
 * Speaking 190 lines takes hours. Without a foreground service Android will
 * suspend the process as soon as the screen locks, and the run silently
 * stalls — the user comes back to a progress bar that has not moved.
 *
 * The notification is not decoration: a foreground service is the only way to
 * be allowed this much background CPU, and it doubles as the progress display
 * when the app is not on screen. A partial wake lock is held alongside it,
 * because a foreground service alone does not stop the CPU from dozing.
 */
class DubbingService : Service() {

    companion object {
        private const val CHANNEL_ID = "vdub_work"
        private const val NOTIFICATION_ID = 1001

        const val EXTRA_TITLE = "title"
        const val EXTRA_TEXT = "text"
        const val EXTRA_PROGRESS = "progress"      // 0..100, -1 = indeterminate

        fun start(context: Context, title: String, text: String) {
            val intent = Intent(context, DubbingService::class.java)
                .putExtra(EXTRA_TITLE, title)
                .putExtra(EXTRA_TEXT, text)
                .putExtra(EXTRA_PROGRESS, -1)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun update(context: Context, title: String, text: String, progress: Int) {
            val intent = Intent(context, DubbingService::class.java)
                .putExtra(EXTRA_TITLE, title)
                .putExtra(EXTRA_TEXT, text)
                .putExtra(EXTRA_PROGRESS, progress)
            context.startService(intent)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, DubbingService::class.java))
        }
    }

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createChannel()
        acquireWakeLock()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val title = intent?.getStringExtra(EXTRA_TITLE) ?: "vdub"
        val text = intent?.getStringExtra(EXTRA_TEXT).orEmpty()
        val progress = intent?.getIntExtra(EXTRA_PROGRESS, -1) ?: -1

        val notification = buildNotification(title, text, progress)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        // Restart if the system kills us mid-run; the pipeline itself is
        // resumable, so picking up again is safe.
        return START_STICKY
    }

    override fun onDestroy() {
        runCatching { wakeLock?.takeIf { it.isHeld }?.release() }
        wakeLock = null
        super.onDestroy()
    }

    private fun acquireWakeLock() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "vdub:dubbing").apply {
            setReferenceCounted(false)
            // Long but bounded: a stuck job must not hold the CPU forever.
            acquire(6 * 60 * 60 * 1000L)
        }
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(CHANNEL_ID) != null) return
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Dubbing progress",
                NotificationManager.IMPORTANCE_LOW    // silent; it updates often
            ).apply {
                description = "Shows progress while vdub is working"
                setShowBadge(false)
            }
        )
    }

    private fun buildNotification(title: String, text: String, progress: Int): Notification {
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_stat_vdub)
            .setContentIntent(open)
            .setOngoing(true)
            .setSilent(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .apply {
                if (progress in 0..100) setProgress(100, progress, false)
                else setProgress(0, 0, true)
            }
            .build()
    }
}
