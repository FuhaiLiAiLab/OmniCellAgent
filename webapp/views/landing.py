from dash import dcc, html
import dash_bootstrap_components as dbc

layout = html.Div([
    dbc.Row([
        # Left Column - Welcome/Info Section (60% of screen)
        dbc.Col([
            html.Div([
                html.Div([
                    html.Img(src='/assets/dash-logo-stripe.svg', style={
                        'height': '200px',
                        'width': 'auto',
                        'margin-bottom': '30px',
                        'filter': 'drop-shadow(0 4px 8px rgba(93, 64, 55, 0.1))'
                    }),
                    html.H1("OmniCellAgent", style={
                        'color': '#2c1810',
                        'font-family': 'Palatino Linotype, Palatino, Book Antiqua, Georgia, serif',
                        'font-weight': '600',
                        'font-size': '48px',
                        'letter-spacing': '2px',
                        'margin-bottom': '15px',
                        'text-align': 'center'
                    }),
                    html.P("AI Co-Scientist for Autonomous Single-Cell Omics Deep Research", style={
                        'color': '#5d4037',
                        'font-family': 'Georgia, Cambria, Times New Roman, serif',
                        'font-size': '18px',
                        'text-align': 'center',
                        'line-height': '1.4'
                    }),
                    html.Hr(style={'border-color': '#d7ccc8', 'margin': '30px 0'}),
                    
                    # Feature highlights
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-cogs", style={'color': '#5d4037', 'margin-right': '10px', 'font-size': '20px'}),
                            html.Span("Advanced Agentic Orchestration Systems", style={'color': '#5d4037', 'font-family': 'Georgia, serif', 'font-size': '16px'})
                        ], style={'margin-bottom': '15px', 'display': 'flex', 'align-items': 'center'}),
                        
                        html.Div([
                            html.I(className="fas fa-database", style={'color': '#5d4037', 'margin-right': '10px', 'font-size': '20px'}),
                            html.Span("Bio-Focused Specialized Database & Foundation Models", style={'color': '#5d4037', 'font-family': 'Georgia, serif', 'font-size': '16px'})
                        ], style={'margin-bottom': '15px', 'display': 'flex', 'align-items': 'center'}),
                        
                        html.Div([
                            html.I(className="fas fa-brain", style={'color': '#5d4037', 'margin-right': '10px', 'font-size': '20px'}),
                            html.Span("Intelligent Research Automation & Discovery", style={'color': '#5d4037', 'font-family': 'Georgia, serif', 'font-size': '16px'})
                        ], style={'margin-bottom': '30px', 'display': 'flex', 'align-items': 'center'})
                    ], style={'text-align': 'left', 'max-width': '450px', 'margin': '0 auto'}),
                    
                    html.Hr(style={'border-color': '#d7ccc8', 'margin': '30px 0'}),
                    html.P("Powered by Advanced AI", style={
                        'color': '#a1887f',
                        'font-family': 'Georgia, serif',
                        'font-size': '14px',
                        'text-align': 'center',
                        'font-style': 'italic'
                    }),
                    
                    # Footer links
                    html.Div([
                        html.A("[lab]", 
                               href="https://fuhailiailab.github.io", 
                               target="_blank",
                               style={
                                   'color': '#7b5e57', 
                                   'text-decoration': 'none', 
                                   'font-family': 'Georgia, serif', 
                                   'font-size': '14px', 
                                   'margin-right': '15px'
                               }),
                        html.A("[github]", 
                               href="https://github.com/FuhaiLiAiLab/OmniCellAgent", 
                               target="_blank",
                               style={
                                   'color': '#7b5e57', 
                                   'text-decoration': 'none', 
                                   'font-family': 'Georgia, serif', 
                                   'font-size': '14px',
                                   'margin-right': '15px'
                               }),
                           html.A("[youtube]", 
                               href="https://www.youtube.com/@FuhaiLiAILab", 
                               target="_blank",
                               style={
                                'color': '#7b5e57', 
                                'text-decoration': 'none', 
                                'font-family': 'Georgia, serif', 
                                'font-size': '14px',
                                'margin-right': '15px'
                               }),
                        html.A("[paper]", 
                               href="https://www.biorxiv.org/content/10.1101/2025.07.31.667797v1", 
                               target="_blank",
                               style={
                                   'color': '#7b5e57', 
                                   'text-decoration': 'none', 
                                   'font-family': 'Georgia, serif', 
                                   'font-size': '14px'
                               })
                    ], style={'text-align': 'center', 'margin-top': '30px'})
                ], style={
                    'text-align': 'center',
                    'padding': '40px 20px',
                    'height': '100%',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'justify-content': 'center'
                })
            ], style={
                'background': 'linear-gradient(135deg, #efebe9 0%, #fafafa 45%, #fff8e1 100%)',
                'height': '100vh',
                'box-shadow': '2px 0 10px rgba(93, 64, 55, 0.1)'
            })
        ], width=7, style={'padding': '0'}),
        
        # Right Column - Call to Action Section (40% of screen)
        dbc.Col([
            html.Div([
                html.Div([
                    html.H3("Welcome to the Future of Research", style={
                        'color': '#5d4037',
                        'font-family': 'Georgia, Cambria, serif',
                        'font-weight': '600',
                        'margin-bottom': '20px',
                        'text-align': 'center'
                    }),
                    html.P("Access powerful AI-driven tools designed specifically for biomedical research.", style={
                        'color': '#8d6e63',
                        'font-family': 'Georgia, serif',
                        'margin-bottom': '40px',
                        'text-align': 'center',
                        'line-height': '1.6'
                    }),
                    
                    html.Div([
                        html.H5("Ready to get started?", style={
                            'color': '#5d4037',
                            'font-family': 'Georgia, serif',
                            'font-weight': '500',
                            'margin-bottom': '30px',
                            'text-align': 'center'
                        }),
                        
                        dcc.Link(
                            html.Button(
                                children='Get Started',
                                style={
                                    'background': 'linear-gradient(145deg, #6b8e9f 0%, #5a7d8e 100%)',
                                    'color': '#ffffff',
                                    'border': 'none',
                                    'border-radius': '4px',
                                    'padding': '15px 40px',
                                    'font-family': 'Georgia, serif',
                                    'font-weight': '500',
                                    'font-size': '16px',
                                    'width': '100%',
                                    'cursor': 'pointer',
                                    'transition': 'all 0.3s ease',
                                    'box-shadow': '0 2px 8px rgba(107, 142, 159, 0.3)'
                                }
                            ),
                            href="/page2",
                            style={'text-decoration': 'none'}
                        ),
                        
                        html.Div([
                            html.Hr(style={'border-color': '#d7ccc8', 'margin': '40px 0 30px 0'}),
                            html.P([
                                "New to the platform? Learn more about our ",
                                html.A("research capabilities", href="https://fuhailiailab.github.io", target="_blank", style={
                                    'color': '#7b5e57',
                                    'text-decoration': 'none'
                                })
                            ], style={
                                'text-align': 'center',
                                'color': '#a1887f',
                                'font-family': 'Georgia, serif',
                                'font-size': '14px',
                                'line-height': '1.6'
                            })
                        ])
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
                'background': '#fafafa',
                'height': '100vh'
            })
        ], width=5, style={'padding': '0'})
    ], className="g-0", style={'margin': '0'}),
], style={
    'height': '100vh',
    'overflow': 'hidden'
})
