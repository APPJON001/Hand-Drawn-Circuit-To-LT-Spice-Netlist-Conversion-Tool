# Database models

from . import db  # Store db instance
from flask_login import UserMixin
#from sqlalchemy.sql import func


# Create a note class in the database
class Note(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    data = db.Column(db.String(10000)) # Note
    #date = db.Column(db.DateTime(timezone=True), nullable=True, default=func.now)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


# Create a user class in the database
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key = True)
    email = db.Column(db.String(150), unique=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    notes = db.relationship('Note') # Allow multiple notes to be stored for each user