from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from webapp import app, User
from flask_login import login_user
from werkzeug.security import check_password_hash


layout = html.Div([
    dbc.Row([
        # Left Column - Logo/Icon Section (50% of screen)
        dbc.Col([
            html.Div([
                html.Div([
                    html.Img(src='/assets/dash-logo-stripe.svg', style={
                        'height': '200px',
                        'width': 'auto',
                        'margin-bottom': '30px',
                        'filter': 'drop-shadow(0 4px 8px rgba(75, 34, 92, 0.1))'
                    }),
                    html.H2("agent.omni-cell.com", style={
                        'color': '#4b225c',
                        'font-family': 'Times New Roman, serif',
                        'font-weight': '600',
                        'margin-bottom': '20px',
                        'text-align': 'center'
                    }),
                    html.P("Multi-Agentic System for Bio-medical Research", style={
                        'color': '#7e57c2',
                        'font-family': 'Times New Roman, serif',
                        'font-size': '18px',
                        'text-align': 'center',
                        'line-height': '1.4'
                    }),
                    html.Hr(style={'border-color': '#e0e0e0', 'margin': '30px 0'}),
                    html.P("Powered by Advanced AI", style={
                        'color': '#8e8e8e',
                        'font-family': 'Times New Roman, serif',
                        'font-size': '14px',
                        'text-align': 'center',
                        'font-style': 'italic'
                    })
                ], style={
                    'text-align': 'center',
                    'padding': '40px 20px',
                    'height': '100%',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'justify-content': 'center'
                })
            ], style={
                'background': 'linear-gradient(135deg, #f3e5f5 0%, #e3f2fd 45%, #ffe8d5 100%)',
                'height': '100vh',
                'box-shadow': '2px 0 10px rgba(75, 34, 92, 0.1)'
            })
        ], width=6, style={'padding': '0'}),
        
        # Right Column - Login Form Section (50% of screen)
        dbc.Col([
            html.Div([
                dcc.Location(id='urlLogin', refresh=True),
                html.Div([
                    html.H3("Welcome Back", style={
                        'color': '#4b225c',
                        'font-family': 'Times New Roman, serif',
                        'font-weight': '600',
                        'margin-bottom': '10px',
                        'text-align': 'center'
                    }),
                    html.P("Please sign in to your account", style={
                        'color': '#8e8e8e',
                        'font-family': 'Times New Roman, serif',
                        'margin-bottom': '40px',
                        'text-align': 'center'
                    }),
                    
                    html.Div([
                        html.Label("Username", style={
                            'color': '#4b225c',
                            'font-family': 'Times New Roman, serif',
                            'font-weight': '500',
                            'margin-bottom': '8px'
                        }),
                        dcc.Input(
                            placeholder='Enter your username',
                            type='text',
                            id='usernameBox',
                            className='form-control',
                            n_submit=0,
                            style={
                                'border-radius': '0',
                                'border': '1px solid #e0e0e0',
                                'padding': '12px 15px',
                                'font-family': 'Times New Roman, serif',
                                'margin-bottom': '20px'
                            }
                        ),
                    ]),
                    
                    html.Div([
                        html.Label("Password", style={
                            'color': '#4b225c',
                            'font-family': 'Times New Roman, serif',
                            'font-weight': '500',
                            'margin-bottom': '8px'
                        }),
                        dcc.Input(
                            placeholder='Enter your password',
                            type='password',
                            id='passwordBox',
                            className='form-control',
                            n_submit=0,
                            style={
                                'border-radius': '0',
                                'border': '1px solid #e0e0e0',
                                'padding': '12px 15px',
                                'font-family': 'Times New Roman, serif',
                                'margin-bottom': '30px'
                            }
                        ),
                    ]),
                    
                    html.Button(
                        children='Sign In',
                        n_clicks=0,
                        type='submit',
                        id='loginButton',
                        style={
                            'background': 'linear-gradient(145deg, #4b225c 0%, #3a2a47 100%)',
                            'color': '#ffffff',
                            'border': 'none',
                            'border-radius': '0',
                            'padding': '12px 40px',
                            'font-family': 'Times New Roman, serif',
                            'font-weight': '500',
                            'font-size': '16px',
                            'width': '100%',
                            'cursor': 'pointer',
                            'transition': 'all 0.3s ease',
                            'box-shadow': '0 2px 8px rgba(75, 34, 92, 0.2)'
                        }
                    ),
                    
                    html.Div([
                        html.Hr(style={'border-color': '#e0e0e0', 'margin': '30px 0 20px 0'}),
                        html.P([
                            "Need help? Contact ",
                            html.A("support", href="mailto:support@omni-cell.com", style={
                                'color': '#4b225c',
                                'text-decoration': 'none'
                            })
                        ], style={
                            'text-align': 'center',
                            'color': '#8e8e8e',
                            'font-family': 'Times New Roman, serif',
                            'font-size': '14px'
                        })
                    ])
                    
                ], style={
                    'padding': '60px 40px',
                    'max-width': '400px',
                    'margin': '0 auto',
                    'height': '100%',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'justify-content': 'center'
                })
            ], style={
                'background': '#ffffff',
                'height': '100vh'
            })
        ], width=6, style={'padding': '0'})
    ], className="g-0", style={'margin': '0'}),
], style={
    'height': '100vh',
    'overflow': 'hidden'
})



################################################################################
# LOGIN BUTTON CLICKED / ENTER PRESSED - REDIRECT TO PAGE1 IF LOGIN DETAILS ARE CORRECT
################################################################################
@app.callback(Output('urlLogin', 'pathname'),
              [Input('loginButton', 'n_clicks'),
              Input('usernameBox', 'n_submit'),
              Input('passwordBox', 'n_submit')],
              [State('usernameBox', 'value'),
               State('passwordBox', 'value')])
def sucess(n_clicks, usernameSubmit, passwordSubmit, username, password):
    print(f"Login attempt: clicks={n_clicks}, user_submit={usernameSubmit}, pass_submit={passwordSubmit}")
    print(f"Credentials: username='{username}', password='{password}'")
    
    if (n_clicks > 0) or (usernameSubmit > 0) or (passwordSubmit > 0):
        if username and password:
            try:
                from webapp import server
                with server.app_context():
                    user = User.query.filter_by(username=username).first()
                    print(f"User found: {user}")
                    if user:
                        password_check = check_password_hash(user.password, password)
                        print(f"Password check result: {password_check}")
                        if password_check:
                            from flask import session
                            from datetime import timedelta
                            session.permanent = True  # Make session permanent
                            login_user(user, remember=True, duration=timedelta(hours=1))  # 1 hour duration
                            print("Login successful, redirecting to /page1")
                            return '/page2'
                        else:
                            print("Password check failed")
                    else:
                        print("User not found")
            except Exception as e:
                print(f"Login error: {e}")
                import traceback
                traceback.print_exc()
    print("Login failed, staying on /login")
    return '/login'


################################################################################
# LOGIN BUTTON CLICKED / ENTER PRESSED - RETURN RED BOXES IF LOGIN DETAILS INCORRECT
################################################################################
@app.callback(Output('usernameBox', 'className'),
              [Input('loginButton', 'n_clicks'),
              Input('usernameBox', 'n_submit'),
              Input('passwordBox', 'n_submit')],
              [State('usernameBox', 'value'),
               State('passwordBox', 'value')])
def update_username_style(n_clicks, usernameSubmit, passwordSubmit, username, password):
    if (n_clicks > 0) or (usernameSubmit > 0) or (passwordSubmit) > 0:
        if username and password:
            try:
                from webapp import server
                with server.app_context():
                    user = User.query.filter_by(username=username).first()
                    if user and check_password_hash(user.password, password):
                        return 'form-control'
                    else:
                        return 'form-control is-invalid'
            except Exception as e:
                print(f"Username validation error: {e}")
                return 'form-control is-invalid'
        else:
            return 'form-control is-invalid'
    else:
        return 'form-control'


################################################################################
# LOGIN BUTTON CLICKED / ENTER PRESSED - RETURN RED BOXES IF LOGIN DETAILS INCORRECT
################################################################################
@app.callback(Output('passwordBox', 'className'),
              [Input('loginButton', 'n_clicks'),
              Input('usernameBox', 'n_submit'),
              Input('passwordBox', 'n_submit')],
              [State('usernameBox', 'value'),
               State('passwordBox', 'value')])
def update_password_style(n_clicks, usernameSubmit, passwordSubmit, username, password):
    if (n_clicks > 0) or (usernameSubmit > 0) or (passwordSubmit) > 0:
        if username and password:
            try:
                from webapp import server
                with server.app_context():
                    user = User.query.filter_by(username=username).first()
                    if user and check_password_hash(user.password, password):
                        return 'form-control'
                    else:
                        return 'form-control is-invalid'
            except Exception as e:
                print(f"Password validation error: {e}")
                return 'form-control is-invalid'
        else:
            return 'form-control is-invalid'
    else:
        return 'form-control'