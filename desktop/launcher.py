def main():
    try:
        import os
        import sys
        os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
        os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
        if '--self-test' in sys.argv:
            from desktop.jev_sidecar.selftest import run
            position = sys.argv.index('--report')
            run(sys.argv[position + 1], model_test='--model-test' in sys.argv)
            return
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
        if sys.platform == 'win32' and '--self-test' not in sys.argv:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, str(error) + '\n\nLog: ' + str(log), 'Jev Mahjong', 0x10)
        raise

if __name__ == '__main__':
    main()
