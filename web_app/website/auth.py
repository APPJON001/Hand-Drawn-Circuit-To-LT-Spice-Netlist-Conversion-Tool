from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
# For user security and password hashing
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from flask_login import login_user, login_required, logout_user, current_user



auth = Blueprint('auth',  __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form.get('username') # Can be username or email
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username_or_email).first() # Check username
        if not user: user = User.query.filter_by(email=username_or_email).first() # Check email
        if user: 
            if check_password_hash(user.password, password):
                flash("Logged in successfully!", category="success")
                login_user(user, remember=True)
                return redirect(url_for('views.home'))
            else:
                flash("Incorrect password. Try again.", category='error')
        else:
            # If neither email nor username exists, throw an error
            flash("Username or email does not exist.", category='error')

    return render_template("login.html", user=current_user)

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        username = request.form.get('username')
        password = request.form.get('password')
        
        username_exists = User.query.filter_by(username=username).first()
        email_exists = User.query.filter_by(email=email).first()
        
        if username_exists: 
            flash("Username already exists.", category='error')
        elif email_exists: 
            flash("Email already exists.", category='error')
        elif len(email) < 4:
            flash("Email must be at least 4 characters.", category="error")
        elif len(first_name) < 2:
            flash("First name must be at least 2 characters.", category="error")
        elif len(username) < 4:
            flash("Username must be at least 4 characters.", category="error")
        elif len(password) < 5:
            flash("Password must be at least 5 characters.", category="error")
        else: 
            new_user = User(email=email, first_name=first_name, username=username, password = generate_password_hash(password, method='sha256')) # Set up a user in db
            # Add user to database
            db.session.add(new_user)
            db.session.commit()
            flash("Account created!", category = "success")
            login_user(new_user, remember=True)
            return redirect(url_for('views.home'))
            
    return render_template("sign_up.html", user=current_user)