import dash_bootstrap_components as dbc
from dash import html

def create_navbar():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.NavItem(dbc.NavLink("Context", href="/context")),
            dbc.NavItem(dbc.NavLink("Charts", href="/charts")),
            dbc.NavItem(dbc.NavLink("Map", href="/map")),
            dbc.NavItem(dbc.NavLink("Prédiction", href="/predict")),
            dbc.NavItem(dbc.NavLink("About", href="/about")),
        ],
        brand="Mon Application Dash",
        brand_href="/",
        color="primary",
        dark=True,
    )
    return navbar
