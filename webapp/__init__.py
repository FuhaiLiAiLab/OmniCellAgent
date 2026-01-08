# Flask Application Factory for bioRAG webapp
import dash
import dash_bootstrap_components as dbc
import os
from flask_login import LoginManager, UserMixin


# Initialize Dash app with proper configuration
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
])

# Set the page title
app.title = "agent.omni-cells"

server = app.server
app.config.suppress_callback_exceptions = True

# Flask server configuration for persistent sessions
server.config['SECRET_KEY'] = 'biorag-secret-key-for-sessions-2024'
server.config['SESSION_COOKIE_SECURE'] = False
server.config['SESSION_COOKIE_HTTPONLY'] = True
server.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
server.config['PERMANENT_SESSION_LIFETIME'] = 3600
server.config['SESSION_USE_SIGNER'] = True
server.config['DEBUG'] = False
server.config['TESTING'] = False

# Configuration for better reverse proxy (ngrok) compatibility
server.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload/download
server.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for downloads
server.config['PREFERRED_URL_SCHEME'] = 'https'  # ngrok uses HTTPS

# Custom CSS for the bioRAG UI
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/svg+xml" href="assets/favicon.svg">
        <link rel="shortcut icon" href="assets/favicon.svg">
        {%css%}
        <link rel="stylesheet" href="assets/flatly-custom.css">
    </head>
    <body class="dash-template">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Configure user management
from .users_mgt import db, User as base
from .config import config

# Configure Flask app for SQLAlchemy
server.config['SQLALCHEMY_DATABASE_URI'] = config.get('database', 'con')
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = '/login'

# Initialize database with the server
db.init_app(server)

# Create tables within app context
with server.app_context():
    db.create_all()

# User loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# User class
class User(UserMixin, base):
    pass
