# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

a = Analysis(
    ['unity_gaze_bridge.py'],
    pathex=[],
    binaries=[],
    datas=collect_data_files('mediapipe') + [('gazefollower/res', 'gazefollower/res')],
    hiddenimports=['SharedMemoryCamera'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='unity_gaze_bridge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='unity_gaze_bridge',
)