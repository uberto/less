# Web UI for LESS (LLM Empowered Semantic Search)

## Project Overview

This document outlines the plan to create a web-based user interface for the LESS document search tool using Elixir and Phoenix LiveView. The web UI will allow users to:

1. Submit search queries through a web interface
2. View search results in a user-friendly format
3. Open and view PDF documents directly in the browser
4. Navigate to specific pages in the documents where search matches were found

## Architecture

The system will consist of:

1. **Elixir/Phoenix Web Application**
   - Provides the user interface using Phoenix LiveView
   - Handles user interactions and displays results in real-time
   - Renders PDF documents using a PDF.js viewer

2. **Integration with Existing Python Tool**
   - The web app will execute the Python search tool as a subprocess
   - Results will be parsed and displayed in the web interface
   - Document paths will be mapped to web-accessible URLs

## Implementation Plan

### 1. Project Setup
- Create a new Phoenix LiveView project
- Set up Docker for development environment
- Configure project structure and dependencies

### 2. Core Features
- Create the main search interface
- Implement search functionality by calling the Python tool
- Parse and display search results
- Implement PDF viewer integration

### 3. User Experience
- Design a clean, responsive UI
- Add real-time search updates using LiveView
- Implement document navigation and highlighting

### 4. Deployment
- Create Docker configuration for production
- Document deployment process

## Technical Details

### Python Integration
The web application will interact with the existing Python tool by:
1. Spawning Python processes to execute searches
2. Parsing the JSON output from the search tool
3. Mapping file paths to web-accessible URLs

### PDF Viewing
PDF documents will be:
1. Served as static files from the web server
2. Displayed using PDF.js in the browser
3. Automatically scrolled to the relevant page based on search results

## Development Roadmap

1. **Phase 1: Basic Setup**
   - Create Phoenix project structure
   - Set up Docker development environment
   - Implement basic UI layout

2. **Phase 2: Search Integration**
   - Create interface to Python search tool
   - Implement search results display
   - Add error handling and validation

3. **Phase 3: PDF Viewing**
   - Integrate PDF.js viewer
   - Implement page navigation
   - Add text highlighting for search terms

4. **Phase 4: Polish & Deployment**
   - Refine UI/UX
   - Optimize performance
   - Create deployment configuration
