from api.app import create_app
from api.config import DevelopmentConfig


application = create_app(
    config_object=DevelopmentConfig)


if __name__ == '__main__':
    
    # Log a message indicating the startup process
    application.logger.info("Starting the Flask application.")
    
    
    application.run(debug=True)
    

    # Log a message indicating the shutdown process
    application.logger.info("Flask application has been shut down.")
