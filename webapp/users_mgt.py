from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))
    admin = db.Column(db.Boolean)


def create_user_table():
    """Create user table - handled by Flask-SQLAlchemy"""
    db.create_all()


def add_user(username, password, email, admin):
    """Add a new user using Flask-SQLAlchemy ORM"""
    hashed_password = generate_password_hash(password)
    
    new_user = User(
        username=username,
        email=email,
        password=hashed_password,
        admin=admin
    )
    
    db.session.add(new_user)
    db.session.commit()


def update_password(username, password):
    """Update user password using Flask-SQLAlchemy ORM"""
    hashed_password = generate_password_hash(password)
    
    user = User.query.filter_by(username=username).first()
    if user:
        user.password = hashed_password
        db.session.commit()


def show_users():
    """Get all users - this will be called from within the app context"""
    try:
        users = User.query.all()
        
        user_list = []
        for user in users:
            user_list.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'admin': str(user.admin)
            })
        
        return user_list
    except RuntimeError:
        # If no app context, return empty list for now
        return []
