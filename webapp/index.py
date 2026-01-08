# webapp/index.py - Main entry point for bioRAG webapp
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import sys
import os

# Add the parent directory to the path to access agent tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from webapp import app, server
from flask_login import logout_user, current_user
from webapp.views import login, error, page1, page2, profile, user_admin, landing


navBar = dbc.Navbar(
    children=[
        dbc.Container([
            dbc.NavbarBrand("agent.omni-cells.com", href="/", className="ms-2"),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav(
                    id='navBar',
                    className="ms-auto",
                    navbar=True,
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ])
    ],
    color="primary",
    dark=True,
    sticky="top",
    className='navbar navbar-expand-lg navbar-dark bg-primary',
)


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.Div(id='navBarContainer'),
        html.Div(id='pageContent')
    ])
], id='table-wrapper')


################################################################################
# HANDLE PAGE ROUTING - SHOW LANDING PAGE FIRST, THEN GO TO PAGE2
################################################################################
@app.callback(Output('pageContent', 'children'),
              [Input('url', 'pathname')])
def displayPage(pathname):
    if pathname == '/':
        return landing.layout  # Show landing page first

    elif pathname == '/logout':
        if current_user.is_authenticated:
            logout_user()
        return landing.layout  # Redirect to landing page after logout

    if pathname == '/page1':
        return page1.layout  # Dashboard

    if pathname == '/page2':
        return page2.layout  # Agent tab (no authentication required)

    if pathname == '/profile':
        # if current_user.is_authenticated:
        return profile.layout
        # else:
        #     return landing.layout

    if pathname == '/admin':
        # if current_user.is_authenticated:
        #     if current_user.admin == 1:
        return user_admin.layout
        #     else:
        #         return error.layout
        # else:
        #     return landing.layout

    if pathname == '/login':
        return page2.layout  # Redirect to Agent tab instead of login page

    if pathname == '/landing':
        return landing.layout  # Landing page

    else:
        return error.layout


################################################################################
# NAVBAR TOGGLER CALLBACK
################################################################################
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

################################################################################
# SHOW/HIDE NAVIGATION BAR - HIDE ON LANDING PAGE
################################################################################
@app.callback(
    Output('navBarContainer', 'children'),
    [Input('pageContent', 'children')])
def showNavBar(input1):
    # Check if we're on the landing page by inspecting the content
    # Landing page should not show navbar
    if input1 == landing.layout:
        return html.Div()
    else:
        return navBar

################################################################################
# SIMPLIFIED NAVIGATION BAR (AGENT ONLY)
################################################################################
@app.callback(
    Output('navBar', 'children'),
    [Input('pageContent', 'children')])
def navBarContent(input1):
    # Show only Agent tab in navbar
    navBarContents = [
        dbc.NavItem(dbc.NavLink('Agent', href='/page2')),
    ]
    return navBarContents



if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=8050)


# To run with gunicorn:
# conda activate autogen-latest && gunicorn --preload -w 1 -b 0.0.0.0:8050 --threads 4 --worker-class=gthread webapp.index:server