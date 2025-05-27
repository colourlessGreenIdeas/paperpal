const express = require('express');
const path = require('path');
const fs = require('fs').promises;
const app = express();

// Serve static files from the current directory
app.use(express.static('.'));

// List files in output directory
app.get('/api/list-files', async (req, res) => {
    try {
        const outputDir = path.join(__dirname, 'output');
        const files = await fs.readdir(outputDir);
        
        // Filter out markdown files and images folder
        const fileList = await Promise.all(
            files
                .filter(filename => {
                    return !filename.endsWith('.md') && 
                           filename !== 'images' &&
                           !filename.startsWith('.');
                })
                .map(async (filename) => {
                    const filePath = path.join(outputDir, filename);
                    const stats = await fs.stat(filePath);
                    
                    let type = 'file';
                    if (filename.endsWith('.json')) type = 'json';
                    else if (filename.endsWith('.pdf')) type = 'pdf';
                    
                    return {
                        name: filename,
                        path: filename,
                        type,
                        size: stats.size,
                        modified: stats.mtime
                    };
                })
        );
        
        // Sort by most recently modified
        fileList.sort((a, b) => new Date(b.modified) - new Date(a.modified));
        
        res.json(fileList);
    } catch (error) {
        console.error('Error listing files:', error);
        res.status(500).json({ error: error.message });
    }
});

// Get file content
app.get('/api/file/:filename', async (req, res) => {
    try {
        const filePath = path.join(__dirname, 'output', req.params.filename);
        // Don't allow access to markdown files or images folder
        if (req.params.filename.endsWith('.md') || req.params.filename === 'images') {
            throw new Error('Access to this file type is not allowed');
        }
        const content = await fs.readFile(filePath, 'utf8');
        res.send(content);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Delete file
app.delete('/api/file/:filename', async (req, res) => {
    try {
        // Don't allow deleting markdown files or images folder
        if (req.params.filename.endsWith('.md') || req.params.filename === 'images') {
            throw new Error('Cannot delete this file type');
        }
        const filePath = path.join(__dirname, 'output', req.params.filename);
        await fs.unlink(filePath);
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
}); 