from src import default_logger, TrainingUI

def main():
    try:
        ui = TrainingUI()
        ui.run()
    except Exception as e:
        default_logger.log_error(e, "main execution")
    finally:
        default_logger.close()

if __name__ == "__main__":
    main()