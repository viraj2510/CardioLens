"""Vercel entrypoint for the CardioLens Flask application."""

from app import create_app

app = create_app()
