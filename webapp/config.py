import configparser
from sqlalchemy import create_engine
import os

config = configparser.ConfigParser()
# Use absolute path relative to this file's location
config_path = os.path.join(os.path.dirname(__file__), 'config.txt')
config.read(config_path)

# Get the database connection string and make the path absolute
db_con = config.get('database', 'con')
if db_con.startswith('sqlite:///'):
    # Convert relative path to absolute path
    db_path = db_con.replace('sqlite:///', '')
    if not os.path.isabs(db_path):
        # Make it relative to the webapp directory
        db_path = os.path.join(os.path.dirname(__file__), db_path.replace('webapp/', ''))
        db_con = f'sqlite:///{os.path.abspath(db_path)}'

engine = create_engine(db_con)

# Store the processed config for use by other modules
class Config:
    def get(self, section, key):
        if section == 'database' and key == 'con':
            return db_con
        return config.get(section, key)

config = Config()