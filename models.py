from database import mongo
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()

class User:
    @staticmethod
    def register_user(name, email, password):
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        user = {"name": name, "email": email, "password": hashed_pw}
        mongo.db.users.insert_one(user)
        return True

    @staticmethod
    def login_user(email, password):
        user = mongo.db.users.find_one({"email": email})
        if user and bcrypt.check_password_hash(user['password'], password):
            return user
        return None
