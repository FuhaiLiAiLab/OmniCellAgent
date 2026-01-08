#!/usr/bin/env python3
"""
Enhanced database setup script that creates multiple users
"""
import os
import sys
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
import configparser

# Add parent directory to path to access webapp module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Read config
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.txt')
config.read(config_path)

# Create Flask app
app = Flask(__name__)
# Build database URI with absolute path for better reliability
db_path = os.path.join(os.path.dirname(__file__), 'instance', 'dash_user_management.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
print(f"Database path: {db_path}")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))
    admin = db.Column(db.Boolean)

def reset_database():
    """Reset the database and create new users"""
    # Get passwords from environment variables
    admin_pass = os.getenv('ADMIN_PASSWORD', None)
    user_pass = os.getenv('USER_PASSWORD', None)
    
    if not admin_pass or not user_pass:
        print("ERROR: Required environment variables not set!")
        print("Please set ADMIN_PASSWORD and USER_PASSWORD environment variables")
        print("\nExample:")
        print("export ADMIN_PASSWORD='YourSecureAdminPassword123!'")
        print("export USER_PASSWORD='YourSecureUserPassword123!'")
        sys.exit(1)
    
    with app.app_context():
        try:
            # Drop all tables if they exist
            db.drop_all()
            print("Dropped all existing tables")
        except Exception as e:
            print(f"No existing tables to drop: {e}")
        
        # Create all tables
        db.create_all()
        print("Created fresh database tables")
        
        # Create admin user
        admin_password = generate_password_hash(admin_pass)
        admin_user = User(
            username='admin',
            email='admin@example.com', 
            password=admin_password,
            admin=True
        )
        
        # Create regular user
        user_password = generate_password_hash(user_pass)
        regular_user = User(
            username='user',
            email='user@example.com', 
            password=user_password,
            admin=False
        )
        

        # Add users to session
        db.session.add(admin_user)
        db.session.add(regular_user)
        
        
        # Commit changes
        db.session.commit()
        
        print("\n" + "="*50)
        print("DATABASE RESET COMPLETE!")
        print("="*50)
        print("Created users:")
        print("1. Admin User:")
        print("   Username: admin")
        print("   Password: (from ADMIN_PASSWORD env var)")
        print("   Admin: Yes")
        print()
        print("2. Regular User:")
        print("   Username: user")
        print("   Password: (from USER_PASSWORD env var)")
        print("   Admin: No")
        print("="*50)
        print("\nIMPORTANT: Remember to:")
        print("  - Change these default users in production")
        print("  - Use strong, unique passwords")
        print("  - Store passwords securely (not in code/env files)")
        print("="*50)

if __name__ == '__main__':
    reset_database()
