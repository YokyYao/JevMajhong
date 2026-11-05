def main():
    try:
        from desktop.jev_sidecar.app import main as run
        run()
    except Exception as error:
        import os
        from pathlib import Path
        import sys
        import traceback
        log = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'JevMahjong' / 'error.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc(), encoding='utf-8')
        if sys.platform == 'win32':
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, str(error) + '\n\nLog: ' + str(log), 'Jev Mahjong', 0x10)
        raise

if __name__ == '__main__':
    main()
