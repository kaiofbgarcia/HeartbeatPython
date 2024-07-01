[app]
title = Heart Rate Monitor
package.name = heartrate
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1
requirements = python3,kivy,opencv-python-headless,numpy,scipy,scikit-learn,matplotlib
orientation = portrait
android.permissions = CAMERA

[buildozer]
log_level = 2
warn_on_root = 1
