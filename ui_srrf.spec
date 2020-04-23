# -*- mode: python -*-

block_cipher = None


a = Analysis(['srrf_cupy\\ui_srrf.py'],
             pathex=['.'],
             binaries=[],
             datas = [],
             hiddenimports=['fastrlock', 'fastrlock.rlock', 'cupy.core.flags', 'srrf_cupy.ui' ],
             hookspath=['srrf_cupy'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='srrf',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='srrf')
