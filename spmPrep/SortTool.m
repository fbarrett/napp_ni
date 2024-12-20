function varargout = SortTool(varargin)
% SORTTOOL MATLAB code for SortTool.fig
%      SORTTOOL, by itself, creates a new SORTTOOL or raises the existing
%      singleton*.
%
%      H = SORTTOOL returns the handle to a new SORTTOOL or the handle to
%      the existing singleton*.
%
%      SORTTOOL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SORTTOOL.M with the given input arguments.
%
%      SORTTOOL('Property','Value',...) creates a new SORTTOOL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SortTool_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SortTool_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SortTool

% Last Modified by GUIDE v2.5 13-Dec-2016 13:04:39

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SortTool_OpeningFcn, ...
                   'gui_OutputFcn',  @SortTool_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before SortTool is made visible.
function SortTool_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SortTool (see VARARGIN)

% Choose default command line output for SortTool
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SortTool wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SortTool_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
%select dirs
function pushbutton1_Callback(hObject, eventdata, handles)
sessions=uipickfiles();
handles.sessions=sessions;
guidata(hObject,handles);
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton2.
%run
function pushbutton2_Callback(hObject, eventdata, handles)
if ~isfield(handles,'sessions')
    error('Did not choose directories');
else
    sessions=handles.sessions;
    sortsPath=fullfile(fileparts(which('sort_nidata')),'*.m');
    file=uigetfile(fullfile(fileparts(which('sort_nidata')),'*.m'));
    cwd=pwd;
    for i=1:length(sessions)
        cd(sessions{i});
        run(file);
    end
    cd(cwd);
end


% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
