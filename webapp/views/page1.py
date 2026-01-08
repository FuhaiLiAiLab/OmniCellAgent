
# Dash packages
import dash_bootstrap_components as dbc
from dash import html

from webapp import app


###############################################################################
########### DASHBOARD PAGE LAYOUT ###########
###############################################################################
layout = dbc.Container([

        html.H2('Dashboard'),
        html.Hr(),


], className="mt-4")
