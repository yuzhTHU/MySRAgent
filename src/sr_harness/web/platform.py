"""Workspace and lifecycle HTTP endpoints for the interactive workbench."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import Body, HTTPException, Request
from fastapi.responses import FileResponse

from .session import InteractiveSession

MAX_UPLOAD = 256 * 1024 * 1024


def mount_platform(app, session: InteractiveSession):
    app.state.session = session

    @app.get('/api/session')
    def status():
        return session.snapshot()

    @app.post('/api/session/start')
    def start(payload: dict = Body(...)):
        try:
            return session.start(payload)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/session/settings')
    def settings(payload: dict = Body(...)):
        try:
            return session.configure(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def resolve(path):
        try:
            return session.resolve(path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/api/workspace')
    def files(path: str = ''):
        with session.lock:
            directory = resolve(path)
            if not directory.is_dir():
                raise HTTPException(404, 'Directory not found')
            entries = []
            for item in directory.iterdir():
                if item.is_symlink():
                    continue
                try:
                    entries.append({'name': item.name, 'path': str(item.relative_to(session.workspace)),
                                    'directory': item.is_dir(), 'size': item.stat().st_size})
                except FileNotFoundError:
                    continue
            return {'path': path, 'entries': sorted(entries, key=lambda e: (not e['directory'], e['name']))}

    @app.get('/api/workspace/download')
    def download(path: str):
        with session.lock:
            file = resolve(path)
            if not file.is_file():
                raise HTTPException(404, 'File not found')
            return FileResponse(file, filename=file.name, media_type='application/octet-stream')

    @app.get('/api/workspace/preview')
    def preview(path: str):
        with session.lock:
            file = resolve(path)
            if not file.is_file():
                raise HTTPException(404, 'File not found')
            with file.open('rb') as stream:
                data = stream.read(65537)
            return {'text': data[:65536].decode('utf-8', errors='replace'), 'truncated': len(data) > 65536}

    @app.get('/api/data/csv-files')
    def csv_files():
        with session.lock:
            files = [
                str(path.relative_to(session.workspace))
                for path in session.workspace.rglob('*')
                if path.is_file() and not path.is_symlink() and path.suffix.lower() == '.csv'
            ]
            return {'files': sorted(files)}

    @app.get('/api/data/preview')
    def data_preview(path: str, rows: int = 300):
        rows = max(5, min(rows, 1000))
        with session.lock:
            file = resolve(path)
            if file.suffix.lower() != '.csv' or not file.is_file():
                raise HTTPException(400, 'Select an existing CSV file')
            try:
                frame = pd.read_csv(file, nrows=rows + 1)
            except Exception as exc:
                raise HTTPException(400, f'Unable to read CSV: {exc}') from exc
        if frame.empty or len(frame.columns) < 2:
            raise HTTPException(400, 'CSV needs at least two columns and one row')
        columns = [str(column) for column in frame.columns]
        if len(set(columns)) != len(columns):
            raise HTTPException(400, 'CSV column names must be unique')
        frame.columns = columns
        truncated = len(frame) > rows
        frame = frame.iloc[:rows]
        numeric = {
            column: bool(pd.api.types.is_numeric_dtype(frame[column]))
            for column in columns
        }
        serializable = frame.astype(object).where(frame.notna(), None)
        serializable = serializable.replace([float('inf'), float('-inf')], None)
        records = serializable.to_dict(orient='records')
        return {'path': path, 'columns': columns, 'numeric': numeric,
                'rows': records, 'truncated': truncated}

    @app.post('/api/data/demo')
    def create_demo():
        path = session.create_demo()
        relative = str(path.relative_to(session.workspace))
        session.controller.publish('file_uploaded', {'path': relative, 'size': path.stat().st_size})
        return {'path': relative}

    @app.post('/api/data/prompts')
    def initial_prompts(payload: dict = Body(...)):
        try:
            return session.preview_initial_prompts(payload)
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put('/api/workspace/upload')
    async def upload(request: Request, path: str):
        # Stream into a staging file, then atomically place it in the *current*
        # workspace (which can move when fit initializes its tools).
        fd, temp = tempfile.mkstemp(prefix='.upload-', dir=session.run_dir)
        size = 0
        try:
            with os.fdopen(fd, 'wb') as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, 'Maximum file size is 256 MiB')
                    output.write(chunk)
            with session.lock:
                destination = resolve(path)
                if destination.exists():
                    raise HTTPException(409, 'File already exists; rename it before uploading')
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temp, destination)
            session.controller.publish('file_uploaded', {'path': path, 'size': size})
            return {'path': path, 'size': size}
        finally:
            Path(temp).unlink(missing_ok=True)
