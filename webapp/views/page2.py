
# Dash packages
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context, no_update as dash_no_update
import uuid
import os
import asyncio
import time
import threading
import json
import psutil
import zipfile
from datetime import datetime, timedelta

from webapp import app

# Import the bioRAG UI components and functions
import sys
from pathlib import Path
# From webapp/views/page2.py, we need to go up 2 levels to reach the root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the agent system
from agent.simple_magentic_agent import SimpleMagneticAgentSystem
from dotenv import load_dotenv
load_dotenv()

# Import the UI system class and helper functions from the webapp UI directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'UI'))
from UI.dash_UI import (
    UIMagneticAgentSystem, 
    create_autogen_team,
    add_message_to_queue,
    add_process_step,
    set_final_result,
    add_left_chat_message,
    set_user_question,
    append_to_final_result,
    get_left_chat_display,
    get_final_results_display,
    get_process_details_display,
    start_processing_in_background,
    stop_processing,
    add_multimedia_message,
    add_multimedia_process_step,
    SESSIONS,
    PENDING_USER_INPUTS,
    MESSAGE_QUEUES,
    LEFT_CHAT_MESSAGES,
    PROCESSING_STATUS,
    FINAL_RESULTS,
    PROCESS_DETAILS,
    STEP_STATES,
    COLLAPSE_STATE,
    BACKGROUND_THREADS,
    SESSIONS_DIR
)

###############################################################################
########### AGENT PAGE LAYOUT - BioRAG UI ###########
###############################################################################
layout = dbc.Container([
    dcc.Store(id='session-id', storage_type='session'),
    dcc.Interval(id='message-updater', interval=3000, n_intervals=0, disabled=False),
    
    # Title Section
    html.Div([
        html.Span("Multi-Agentic System for Single-Cell Omics Research", 
                 style={'color': '#ffb74d', 'font-size': '30px', 'font-weight': '600', 'font-family': 'Times New Roman, serif'}),
        html.Span([
            html.Span(" by "),
            html.A("FuhaiLiAiLab", 
                   href="https://profiles.wustl.edu/en/persons/fuhai-li", 
                   target="_blank",
                   style={'color': '#4b225c', 'text-decoration': 'none'}),
            html.Span(" | "),
            html.A("GitHub", 
                   href="https://github.com/fuhailiailab", 
                   target="_blank",
                   style={'color': '#4b225c', 'text-decoration': 'none'})
        ], style={'color': '#4b225c', 'font-size': '22px', 'font-style': 'italic', 'font-family': 'Times New Roman, serif', 'margin-left': '20px'})
    ], style={'display': 'flex', 'justify-content': 'flex-start', 'align-items': 'center', 'margin': '20px 0 15px 0'}),

    # Main Content Area
    dbc.Row([
        # Left Panel - Final Results
        dbc.Col([
            html.Div([
                html.H4("📝 Final Results", style={'margin-bottom': '15px', 'color': '#4b225c'}),
                html.Div(id='final-results-content', children=[
                    html.P("Results will appear here after processing...", 
                          style={'color': '#6c757d', 'font-style': 'italic'})
                ])
            ], className='results-panel')
        ], width=6),
        
        # Right Panel - Step-wise Process Details
        dbc.Col([
            html.Div([
                html.Div([
                    html.H4("🔧 Step-wise Process Details", style={'margin-bottom': '15px', 'color': '#4b225c', 'display': 'inline-block'}),
                    html.Div([
                        dbc.Button("Expand All", id='expand-all-button', n_clicks=0, 
                                  color='light', size='sm', 
                                  style={'margin-right': '5px', 'border-radius': '0', 'font-size': '0.8rem'}),
                        dbc.Button("Collapse All", id='collapse-all-button', n_clicks=0, 
                                  color='light', size='sm', 
                                  style={'border-radius': '0', 'font-size': '0.8rem'})
                    ], style={'float': 'right', 'margin-top': '10px'})
                ], style={'overflow': 'hidden', 'margin-bottom': '15px'}),
                html.Div(id='process-details-content', children=[
                    html.P("Process steps will appear here during execution...", 
                          style={'color': '#6c757d', 'font-style': 'italic'})
                ])
            ], className='process-panel')
        ], width=6)
    ], style={'margin-bottom': '20px'}),
    
    # Input Section with Team Config
    html.Div([
        dbc.Row([
            # Team Configuration
            dbc.Col([
                html.Div([
                    html.Label("Team Configuration:", style={'font-weight': 'bold', 'margin-bottom': '8px', 'font-size': '0.9rem'}),
                    dcc.RadioItems(
                        id='team-type-selector',
                        options=[
                            # {'label': ' Simple Assistant', 'value': 'simple'},
                            {'label': ' Biomedical Research Team', 'value': 'magentic'}
                        ],
                        value='magentic',
                        inline=False,
                        style={'font-size': '0.85rem'}
                    )
                ], className='team-config')
            ], width=3),
            
            # Text Input
            dbc.Col([
                dcc.Textarea(
                    id='user-input',
                    placeholder='Enter your bio-medical research question or task...',
                    style={
                        'width': '100%', 
                        'height': '120px',
                        'resize': 'vertical',
                        'border': '1px solid #e0e0e0',
                        'border-radius': '0',
                        'padding': '10px',
                        'font-family': 'Times New Roman, Times, serif',
                        'color': '#4b225c'
                    }
                )
            ], width=7),
            
            # Action Buttons
            dbc.Col([
                dbc.Button('Send', id='send-button', n_clicks=0, 
                          color='primary', size='lg', 
                          style={'width': '100%', 'height': '40px', 'margin-bottom': '5px', 'border-radius': '0'}),
                dbc.Button('Stop', id='stop-button', n_clicks=0, 
                          color='danger', size='sm', 
                          style={'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0'}),
                dbc.Button('New Session', id='new-chat-button', n_clicks=0, 
                          color='secondary', size='sm', 
                          style={'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0'}),
                dbc.Button('📥 Download Session', id='download-session-button', n_clicks=0, 
                          color='success', size='sm', 
                          style={'width': '100%', 'height': '35px', 'border-radius': '0'}),
                dcc.Download(id='download-session-zip')
            ], width=2)
        ])
    ], className='input-section')
    
], fluid=True, style={'max-width': '1400px'}, className="mt-4")

###############################################################################
########### AGENT PAGE CALLBACKS - BioRAG UI Functionality ###########
###############################################################################

# --- Initialize Session on Page Load ---
@app.callback(
    Output('session-id', 'data', allow_duplicate=True),
    Input('session-id', 'id'),  # Triggers when the store component is created
    prevent_initial_call='initial_duplicate'
)
def initialize_session_on_load(_):
    """Initialize session when page first loads."""
    session_id = str(uuid.uuid4())
    print(f"[PAGE LOAD] Initializing new session: {session_id}")
    
    # Initialize session data
    MESSAGE_QUEUES[session_id] = []
    PROCESSING_STATUS[session_id] = {"status": "idle", "task_id": None}
    FINAL_RESULTS[session_id] = ""
    LEFT_CHAT_MESSAGES[session_id] = []
    PROCESS_DETAILS[session_id] = []
    
    # Add welcome messages
    add_process_step(session_id, "System", "Welcome to the Multi-Agentic System for Bio-medical Research! Your session is ready.")
    add_left_chat_message(session_id, "System", "Welcome! I'm ready to help with your biomedical research questions.", "system")
    
    return {'id': session_id}

# --- Message Update Callback (Real-time) ---
@app.callback(
    [Output('final-results-content', 'children'),
     Output('process-details-content', 'children')],
    Input('message-updater', 'n_intervals'),
    State('session-id', 'data'),
    State('process-details-content', 'children'),  # Add current state
    prevent_initial_call=True
)
def update_display_panels(n_intervals, session_data, current_process_content):
    """Update both chat messages and process details panels."""
    if not session_data or 'id' not in session_data:
        return ([html.P("Your conversation will appear here...", 
                       style={'color': '#6c757d', 'font-style': 'italic', 'text-align': 'center', 'margin-top': '50px'})],
                [html.P("Process steps will appear here during execution...", 
                       style={'color': '#6c757d', 'font-style': 'italic'})])
    
    session_id = session_data['id']
    
    # Get chat messages for left panel
    chat_messages = get_left_chat_display(session_id)
    
    # Check if we have new process steps
    current_step_count = len(PROCESS_DETAILS.get(session_id, []))
    
    # If we have current content and the step count hasn't changed, don't update process details
    if (current_process_content and 
        current_process_content != [html.P("Process steps will appear here during execution...", 
                                         style={'color': '#6c757d', 'font-style': 'italic'})] and
        hasattr(current_process_content, '__len__') and
        len(current_process_content) == current_step_count):
        return chat_messages, dash_no_update
    
    # Get process details for right panel only if there are new steps
    process_details = get_process_details_display(session_id)
    
    return chat_messages, process_details

# --- Combined Callback for Session and Chat Logic ---
@app.callback(
    [Output('session-id', 'data'),
     Output('user-input', 'value'),
     Output('stop-button', 'style'),
     Output('stop-button', 'disabled')],
    [Input('send-button', 'n_clicks'),
     Input('new-chat-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')],
    [State('user-input', 'value'),
     State('session-id', 'data'),
     State('team-type-selector', 'value')],
    prevent_initial_call=True
)
def handle_user_actions(send_clicks, new_chat_clicks, stop_clicks, user_input, session_data, team_type):
    """Handle user actions (send message, new chat, stop processing) - non-blocking."""
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    session_id = session_data.get('id') if session_data else None

    # Default stop button style (disabled and grayed out)
    stop_button_style = {'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0', 'opacity': '0.5'}
    stop_button_disabled = True

    # --- ACTION 1: Stop Processing ---
    if triggered_id == 'stop-button' and stop_clicks > 0 and session_id:
        stop_processing(session_id)
        return ({'id': session_id}, dash_no_update, stop_button_style, stop_button_disabled)

    # --- ACTION 2: Start a New Chat ---
    if triggered_id == 'new-chat-button' and new_chat_clicks > 0:
        if session_id:
            # Stop any running processing first
            if session_id in BACKGROUND_THREADS:
                stop_processing(session_id)
            
            # Clean up old session
            print(f"Destroying session: {session_id}")
            PENDING_USER_INPUTS.pop(session_id, None)
            SESSIONS.pop(session_id, None)
            MESSAGE_QUEUES.pop(session_id, None)
            PROCESSING_STATUS.pop(session_id, None)
            FINAL_RESULTS.pop(session_id, None)
            LEFT_CHAT_MESSAGES.pop(session_id, None)
            PROCESS_DETAILS.pop(session_id, None)
            BACKGROUND_THREADS.pop(session_id, None)

        new_session_id = str(uuid.uuid4())
        print(f"Creating new session: {new_session_id}")
        
        # Initialize new session
        MESSAGE_QUEUES[new_session_id] = []
        PROCESSING_STATUS[new_session_id] = {"status": "idle", "task_id": None}
        FINAL_RESULTS[new_session_id] = ""
        LEFT_CHAT_MESSAGES[new_session_id] = []
        PROCESS_DETAILS[new_session_id] = []
        
        # Add welcome message to process details
        add_process_step(new_session_id, "System", "New chat session started. Ready to assist with your bio-medical research questions.")
        
        # Add welcome message to left chat
        add_left_chat_message(new_session_id, "System", "Welcome! I'm ready to help with your biomedical research questions.", "system")

        return ({'id': new_session_id}, '', stop_button_style, stop_button_disabled)

    # --- ACTION 3: Send a Message ---
    if triggered_id == 'send-button' and send_clicks > 0 and user_input:
        if not session_id:
            # Create new session if none exists
            session_id = str(uuid.uuid4())
            MESSAGE_QUEUES[session_id] = []
            PROCESSING_STATUS[session_id] = {"status": "idle", "task_id": None}
            FINAL_RESULTS[session_id] = ""
            LEFT_CHAT_MESSAGES[session_id] = []
            PROCESS_DETAILS[session_id] = []
        
        # Add user question to left chat immediately
        set_user_question(session_id, user_input)
        
        # Add user input to process details
        # add_process_step(session_id, "User", f"**Question:** {user_input}")
        
        # Start background processing
        start_processing_in_background(session_id, user_input, team_type)
        
        # Enable stop button
        stop_button_style = {'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0', 'opacity': '1.0'}
        stop_button_disabled = False

        return ({'id': session_id}, '', stop_button_style, stop_button_disabled)

    # No default session creation needed - handled by initialization callback
    return (dash_no_update, dash_no_update, dash_no_update, dash_no_update)

# --- Update Stop Button Visibility Based on Processing Status ---
@app.callback(
    [Output('stop-button', 'style', allow_duplicate=True),
     Output('stop-button', 'disabled', allow_duplicate=True)],
    Input('message-updater', 'n_intervals'),
    State('session-id', 'data'),
    prevent_initial_call=True
)
def update_stop_button_visibility(n_intervals, session_data):
    """Enable/disable stop button based on processing status."""
    if not session_data or 'id' not in session_data:
        return ({'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0', 'opacity': '0.5'}, True)
    
    session_id = session_data['id']
    
    # Check if processing
    is_processing = (
        session_id in PROCESSING_STATUS and 
        PROCESSING_STATUS[session_id].get("status") == "processing" and
        session_id in BACKGROUND_THREADS
    )
    
    if is_processing:
        return ({'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0', 'opacity': '1.0'}, False)
    else:
        return ({'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0', 'opacity': '0.5'}, True)

# --- Expand All / Collapse All Callbacks ---
@app.callback(
    Output('process-details-content', 'children', allow_duplicate=True),
    [Input('expand-all-button', 'n_clicks'),
     Input('collapse-all-button', 'n_clicks')],
    State('session-id', 'data'),
    prevent_initial_call=True
)
def handle_expand_collapse_all(expand_clicks, collapse_clicks, session_data):
    """Handle expand all and collapse all button clicks."""
    if not session_data or 'id' not in session_data:
        return dash.no_update
    
    session_id = session_data['id']
    ctx = callback_context
    
    if not ctx.triggered:
        return dash.no_update
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if session_id not in PROCESS_DETAILS or not PROCESS_DETAILS[session_id]:
        return dash.no_update
    
    # Create new step cards with all expanded or all collapsed
    steps = []
    expand_all = triggered_id == 'expand-all-button' and expand_clicks > 0
    
    for i, step in enumerate(PROCESS_DETAILS[session_id]):
        step_index = f"{session_id}-{i}"
        is_expanded = expand_all  # Expand all or collapse all
        
        step_card = dbc.Card([
            dbc.CardHeader([
                dbc.Button([
                    html.Span("▼ " if is_expanded else "▶ ", 
                             id={'type': 'icon', 'index': step_index}, 
                             style={'margin-right': '5px'}),
                    html.Span(f"{step['agent']}", style={'font-weight': 'bold'}),
                    html.Span(f" - Step {step['step_number']}", style={'color': '#6c757d', 'font-size': '0.9em', 'margin-left': '10px'})
                ], 
                id={'type': 'toggle', 'index': step_index},
                color="light",
                style={
                    'width': '100%', 
                    'text-align': 'left', 
                    'border': 'none',
                    'background': 'transparent',
                    'padding': '10px 15px'
                })
            ], 
            style={'padding': '0', 'background': '#f1f3f4', 'border-bottom': '1px solid #e3e6ea'}),
            
            dbc.Collapse([
                dbc.CardBody([
                    dcc.Markdown(
                        children=step['content'],
                        className='markdown-content',
                        style={'margin': '0'}
                    )
                ], style={'padding': '15px'})
            ], 
            id={'type': 'collapse', 'index': step_index},
            is_open=is_expanded)
        ], style={'margin-bottom': '10px', 'border': '1px solid #e3e6ea'})
        
        steps.append(step_card)
    
    return steps

# --- Toggle Collapsible Process Steps ---
@app.callback(
    [Output({'type': 'collapse', 'index': dash.dependencies.MATCH}, 'is_open'),
     Output({'type': 'icon', 'index': dash.dependencies.MATCH}, 'children')],
    Input({'type': 'toggle', 'index': dash.dependencies.MATCH}, 'n_clicks'),
    State({'type': 'collapse', 'index': dash.dependencies.MATCH}, 'is_open'),
    prevent_initial_call=True
)
def toggle_collapse(n_clicks, is_open):
    """Toggle collapsible sections in the process details panel."""
    if n_clicks:
        new_state = not is_open
        icon = "▼ " if new_state else "▶ "
        return new_state, icon
    return is_open, "▶ "


# --- Download Session Storage ---
# Track zip preparation status per session
DOWNLOAD_STATUS = {}  # session_id -> {"status": "idle/preparing/ready/error", "zip_path": None, "error": None}

# Maximum file size to include in zip (50MB) - skip large binary files
MAX_FILE_SIZE_FOR_ZIP = 50 * 1024 * 1024  # 50MB
# Specific large files to always skip (already compressed to .lrz)
SKIP_FILES = {'expression_matrix.npy'}
# Extensions to always skip (large binary data files)
SKIP_EXTENSIONS = {'.h5', '.hdf5', '.pkl', '.pickle', '.parquet', '.feather'}

def prepare_zip_in_background(session_id, session_dir):
    """Prepare zip file in background thread with size limits."""
    try:
        DOWNLOAD_STATUS[session_id] = {"status": "preparing", "zip_path": None, "error": None}
        print(f"[DEBUG] Starting zip preparation for session: {session_id}")
        
        # Define the zip file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{session_id}_{timestamp}.zip"
        zip_path = os.path.join(session_dir, zip_filename)
        
        # Count and zip files
        file_count = 0
        skipped_files = []
        
        # Use ZIP_DEFLATED with max compression (level 9) - smaller file for transport limits
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for root, dirs, files in os.walk(session_dir):
                for file in files:
                    # Skip session download zip files (but include .npy.zip and .lrz compressed data files)
                    if file.endswith('.zip') and not file.endswith('.npy.zip'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, session_dir)
                    
                    # Check if this specific file should be skipped
                    if file in SKIP_FILES:
                        skipped_files.append(f"{arcname} (large matrix - use .lrz)")
                        print(f"[DEBUG] Skipping large matrix file: {arcname}")
                        continue
                    
                    # Check file extension - skip large binary formats
                    _, ext = os.path.splitext(file.lower())
                    if ext in SKIP_EXTENSIONS:
                        skipped_files.append(f"{arcname} (binary data)")
                        print(f"[DEBUG] Skipping binary file: {arcname}")
                        continue
                    
                    # Check file size
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size > MAX_FILE_SIZE_FOR_ZIP:
                            skipped_files.append(f"{arcname} ({file_size / (1024*1024):.1f}MB)")
                            print(f"[DEBUG] Skipping large file: {arcname} ({file_size / (1024*1024):.1f}MB)")
                            continue
                        
                        zip_file.write(file_path, arcname)
                        file_count += 1
                        print(f"[DEBUG] Added to zip: {arcname}")
                        
                    except OSError as e:
                        print(f"[DEBUG] Error accessing file {arcname}: {e}")
                        skipped_files.append(f"{arcname} (access error)")
                        continue
        
        print(f"[DEBUG] Zip ready: {zip_path} with {file_count} files")
        if skipped_files:
            print(f"[DEBUG] Skipped {len(skipped_files)} files: {skipped_files[:5]}...")
        
        DOWNLOAD_STATUS[session_id] = {"status": "ready", "zip_path": zip_path, "zip_filename": zip_filename, "error": None}
        
    except Exception as e:
        print(f"[ERROR] Failed to create zip: {e}")
        import traceback
        traceback.print_exc()
        DOWNLOAD_STATUS[session_id] = {"status": "error", "zip_path": None, "error": str(e)}


# --- Download Session Zip Callback ---
@app.callback(
    Output('download-session-zip', 'data'),
    Output('download-session-button', 'children'),
    Input('download-session-button', 'n_clicks'),
    Input('message-updater', 'n_intervals'),
    State('session-id', 'data'),
    prevent_initial_call=True
)
def download_session_zip(n_clicks, n_intervals, session_data):
    """Handle download button click and zip preparation status. Auto-downloads when ready."""
    ctx = callback_context
    if not ctx.triggered:
        return None, "📥 Download Session"
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not session_data or 'id' not in session_data:
        return None, "📥 Download Session"
    
    session_id = session_data['id']
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    
    # Check current download status
    current_status = DOWNLOAD_STATUS.get(session_id, {"status": "idle"})
    
    # Auto-download when zip is ready (triggered by interval check)
    if current_status.get("status") == "ready" and current_status.get("zip_path"):
        zip_path = current_status["zip_path"]
        zip_filename = current_status.get("zip_filename", os.path.basename(zip_path))
        if os.path.exists(zip_path):
            # Reset status for next download
            DOWNLOAD_STATUS[session_id] = {"status": "idle", "zip_path": None, "error": None}
            print(f"[DEBUG] Auto-downloading zip: {zip_filename}")
            return dcc.send_file(zip_path, filename=zip_filename), "📥 Download Session"
    
    # If button was clicked, start preparation
    if triggered_id == 'download-session-button' and n_clicks:
        # If not already preparing, start preparation
        if current_status.get("status") not in ["preparing", "ready"]:
            # Ensure session directory exists
            if not os.path.exists(session_dir):
                os.makedirs(session_dir, exist_ok=True)
            
            # Start background thread to prepare zip
            thread = threading.Thread(
                target=prepare_zip_in_background,
                args=(session_id, session_dir),
                daemon=True
            )
            thread.start()
            return None, "⏳ Preparing..."
    
    # Update button text based on status
    if current_status.get("status") == "preparing":
        return None, "⏳ Preparing..."
    elif current_status.get("status") == "error":
        return None, "❌ Error - Try Again"
    
    return None, "📥 Download Session"