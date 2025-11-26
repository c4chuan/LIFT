"""
Prompt Visualization Module

Generates HTML visualizations of prompts and responses during reasoning generation.
Captures all attempts including retries, validation results, and GPT-4o feedback.
"""

import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from datetime import datetime


class PromptRecord:
    """Records a single reasoning generation attempt"""

    def __init__(
        self,
        attempt_number: int,
        messages: List[Dict[str, Any]],
        response: str,
        extracted_action: str,
        validation_passed: bool,
        ground_truth_action: str,
        gpt4o_feedback: Optional[str] = None
    ):
        """
        Initialize a prompt record

        Args:
            attempt_number: Attempt number (1, 2, 3)
            messages: Complete message list sent to model
            response: Model's response (reasoning)
            extracted_action: Action extracted from response
            validation_passed: Whether validation passed
            ground_truth_action: Correct action
            gpt4o_feedback: GPT-4o error analysis feedback (if applicable)
        """
        self.attempt_number = attempt_number
        self.messages = messages
        self.response = response
        self.extracted_action = extracted_action
        self.validation_passed = validation_passed
        self.ground_truth_action = ground_truth_action
        self.gpt4o_feedback = gpt4o_feedback
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ActionPromptHistory:
    """Records all attempts for a single action"""

    def __init__(self, action_index: int):
        """
        Initialize action prompt history

        Args:
            action_index: Index of the action in trajectory
        """
        self.action_index = action_index
        self.records: List[PromptRecord] = []

    def add_record(self, record: PromptRecord):
        """Add a prompt record"""
        self.records.append(record)

    def is_final_success(self) -> bool:
        """Check if final attempt succeeded"""
        return len(self.records) > 0 and self.records[-1].validation_passed


class PromptVisualizer:
    """Generates HTML visualization of prompts and responses"""

    def __init__(self, output_dir: str = "data/prompt_visualizations"):
        """
        Initialize visualizer

        Args:
            output_dir: Output directory for HTML files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html(
        self,
        env_name: str,
        trajectory_name: str,
        intent: str,
        action_histories: List[ActionPromptHistory]
    ) -> str:
        """
        Generate HTML visualization for a trajectory

        Args:
            env_name: Environment name
            trajectory_name: Trajectory file name
            intent: Task intent/objective
            action_histories: List of action prompt histories

        Returns:
            Path to generated HTML file
        """
        # Create environment subdirectory
        env_dir = self.output_dir / env_name
        env_dir.mkdir(parents=True, exist_ok=True)

        # Generate HTML file path
        html_name = trajectory_name.replace('.pkl.xz', '_prompts.html')
        html_path = env_dir / html_name

        # Generate HTML content
        html_content = self._build_html(env_name, trajectory_name, intent, action_histories)

        # Save HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(html_path)

    def _build_html(
        self,
        env_name: str,
        trajectory_name: str,
        intent: str,
        action_histories: List[ActionPromptHistory]
    ) -> str:
        """Build complete HTML content"""

        # Build header
        header = self._build_header(env_name, trajectory_name, intent, action_histories)

        # Build CSS
        css = self._build_css()

        # Build JavaScript
        js = self._build_javascript()

        # Build action sections
        action_sections = ""
        for history in action_histories:
            action_sections += self._build_action_section(history)

        # Combine all parts
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt Visualization - {trajectory_name}</title>
    <style>{css}</style>
</head>
<body>
    {header}
    <div class="container">
        {action_sections}
    </div>
    <script>{js}</script>
</body>
</html>"""

        return html

    def _build_header(
        self,
        env_name: str,
        trajectory_name: str,
        intent: str,
        action_histories: List[ActionPromptHistory]
    ) -> str:
        """Build page header"""

        # Calculate statistics
        total_actions = len(action_histories)
        success_first = sum(1 for h in action_histories if len(h.records) == 1 and h.is_final_success())
        success_retry = sum(1 for h in action_histories if len(h.records) > 1 and h.is_final_success())
        failed = sum(1 for h in action_histories if not h.is_final_success())

        return f"""
<div class="header">
    <h1>Prompt Visualization</h1>
    <div class="metadata">
        <div class="metadata-item"><strong>Environment:</strong> {env_name}</div>
        <div class="metadata-item"><strong>Trajectory:</strong> {trajectory_name}</div>
        <div class="metadata-item"><strong>Intent:</strong> {intent}</div>
    </div>
    <div class="statistics">
        <div class="stat-item stat-total">Total Actions: {total_actions}</div>
        <div class="stat-item stat-success">First Attempt Success: {success_first}</div>
        <div class="stat-item stat-retry">Success After Retry: {success_retry}</div>
        <div class="stat-item stat-fail">Failed: {failed}</div>
    </div>
</div>"""

    def _build_action_section(self, history: ActionPromptHistory) -> str:
        """Build HTML section for one action's attempts"""

        # Determine status class
        if history.is_final_success():
            if len(history.records) == 1:
                status_class = "status-success"
                status_text = "Success (First Attempt)"
            else:
                status_class = "status-retry"
                status_text = f"Success (After {len(history.records) - 1} Retries)"
        else:
            status_class = "status-fail"
            status_text = "Failed"

        # Build attempt sections
        attempts_html = ""
        for record in history.records:
            attempts_html += self._build_attempt_section(record)

        return f"""
<div class="action-section">
    <div class="action-header">
        <h2>Action #{history.action_index + 1}</h2>
        <span class="status-badge {status_class}">{status_text}</span>
    </div>
    {attempts_html}
</div>"""

    def _build_attempt_section(self, record: PromptRecord) -> str:
        """Build HTML for a single attempt"""

        # Validation badge
        if record.validation_passed:
            validation_badge = '<span class="validation-badge validation-pass">✓ Validation Passed</span>'
        else:
            validation_badge = '<span class="validation-badge validation-fail">✗ Validation Failed</span>'

        # Action comparison
        action_comparison = f"""
<div class="action-comparison">
    <div class="action-item">
        <strong>Generated Action:</strong>
        <code class="{'action-correct' if record.validation_passed else 'action-wrong'}">{self._escape_html(record.extracted_action)}</code>
    </div>
    <div class="action-item">
        <strong>Ground Truth:</strong>
        <code class="action-correct">{self._escape_html(record.ground_truth_action)}</code>
    </div>
</div>"""

        # GPT-4o feedback section (if exists)
        gpt4o_section = ""
        if record.gpt4o_feedback:
            gpt4o_section = f"""
<div class="gpt4o-feedback">
    <div class="section-title" onclick="toggleSection(this)">
        <span class="toggle-icon">▼</span> GPT-4o Error Analysis
    </div>
    <div class="section-content">
        <pre>{self._escape_html(record.gpt4o_feedback)}</pre>
    </div>
</div>"""

        # Build messages section
        messages_html = self._build_messages_section(record.messages)

        # Build response section
        response_html = f"""
<div class="response-section">
    <div class="section-title" onclick="toggleSection(this)">
        <span class="toggle-icon">▼</span> Model Response (Reasoning)
    </div>
    <div class="section-content">
        <pre>{self._escape_html(record.response)}</pre>
    </div>
</div>"""

        return f"""
<div class="attempt-section">
    <div class="attempt-header">
        <h3>Attempt #{record.attempt_number}</h3>
        {validation_badge}
        <span class="timestamp">{record.timestamp}</span>
    </div>
    {action_comparison}
    {gpt4o_section}
    {messages_html}
    {response_html}
</div>"""

    def _build_messages_section(self, messages: List[Dict[str, Any]]) -> str:
        """Build messages section with collapsible parts"""

        messages_html = ""

        for idx, message in enumerate(messages):
            role = message.get('role', 'unknown')
            content = message.get('content', '')

            # Determine message type and title
            if role == 'system':
                title = "System Prompt"
                role_class = "role-system"
            elif role == 'user':
                # Try to identify message type by content
                if idx == 1 and isinstance(content, str) and 'Example' in content:
                    title = "Few-Shot Examples"
                    role_class = "role-examples"
                elif isinstance(content, list):
                    title = "Current Query (Multimodal)"
                    role_class = "role-query"
                else:
                    title = f"User Message #{idx}"
                    role_class = "role-user"
            elif role == 'assistant':
                title = "History Reasoning"
                role_class = "role-assistant"
            else:
                title = f"Message #{idx}"
                role_class = "role-unknown"

            # Build message content HTML
            if isinstance(content, str):
                content_html = f'<pre>{self._escape_html(content)}</pre>'
            elif isinstance(content, list):
                # Multimodal content
                content_html = self._build_multimodal_content(content)
            else:
                content_html = f'<pre>{self._escape_html(str(content))}</pre>'

            messages_html += f"""
<div class="message-section {role_class}">
    <div class="section-title" onclick="toggleSection(this)">
        <span class="toggle-icon">▼</span> {title}
    </div>
    <div class="section-content">
        {content_html}
    </div>
</div>"""

        return f"""
<div class="messages-container">
    <div class="section-title" onclick="toggleSection(this)">
        <span class="toggle-icon">▼</span> Complete Message History ({len(messages)} messages)
    </div>
    <div class="section-content">
        {messages_html}
    </div>
</div>"""

    def _build_multimodal_content(self, content_list: List[Dict[str, Any]]) -> str:
        """Build HTML for multimodal content (text + images)"""

        html_parts = []

        for item in content_list:
            item_type = item.get('type', '')

            if item_type == 'text':
                text = item.get('text', '')
                html_parts.append(f'<pre>{self._escape_html(text)}</pre>')

            elif item_type == 'image_url':
                image_url = item.get('image_url', {})
                if isinstance(image_url, dict):
                    url = image_url.get('url', '')
                else:
                    url = image_url

                # Generate unique ID for modal
                img_id = hashlib.md5(url.encode()).hexdigest()[:12]

                html_parts.append(f"""
<div class="image-container">
    <img src="{url}" alt="Screenshot" class="screenshot-thumbnail" onclick="openImageModal('{img_id}')">
    <div id="modal-{img_id}" class="image-modal" onclick="closeImageModal('{img_id}')">
        <span class="close-modal">&times;</span>
        <img src="{url}" class="modal-image">
    </div>
</div>""")

            elif item_type == 'image':
                # PIL Image object - convert to base64 PNG
                image = item.get('image')
                if image is not None:
                    base64_str = self._image_to_base64_png(image)
                    img_id = hashlib.md5(base64_str.encode()).hexdigest()[:12]
                    data_url = f"data:image/png;base64,{base64_str}"

                    html_parts.append(f"""
<div class="image-container">
    <img src="{data_url}" alt="Screenshot" class="screenshot-thumbnail" onclick="openImageModal('{img_id}')">
    <div id="modal-{img_id}" class="image-modal" onclick="closeImageModal('{img_id}')">
        <span class="close-modal">&times;</span>
        <img src="{data_url}" class="modal-image">
    </div>
</div>""")

        return '<div class="multimodal-content">' + ''.join(html_parts) + '</div>'

    @staticmethod
    def _image_to_base64_png(image: Image.Image) -> str:
        """
        Convert PIL Image to base64 PNG string (lossless, no resize)

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded PNG string
        """
        # Convert to RGB if RGBA (PNG supports both, but RGB is more compatible)
        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        elif image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        # Save to bytes as PNG (lossless, no compression level parameter needed)
        buffered = BytesIO()
        image.save(buffered, format="PNG")

        # Encode to base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))

    def _build_css(self) -> str:
        """Build CSS styles"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }

        .header {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        .header h1 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 2em;
        }

        .metadata {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }

        .metadata-item {
            margin: 8px 0;
            font-size: 0.95em;
        }

        .statistics {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .stat-item {
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9em;
        }

        .stat-total {
            background: #e3f2fd;
            color: #1976d2;
        }

        .stat-success {
            background: #e8f5e9;
            color: #388e3c;
        }

        .stat-retry {
            background: #fff3e0;
            color: #f57c00;
        }

        .stat-fail {
            background: #ffebee;
            color: #d32f2f;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .action-section {
            background: white;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .action-header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .action-header h2 {
            font-size: 1.5em;
        }

        .status-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }

        .status-success {
            background: #4caf50;
            color: white;
        }

        .status-retry {
            background: #ff9800;
            color: white;
        }

        .status-fail {
            background: #f44336;
            color: white;
        }

        .attempt-section {
            padding: 20px;
            border-bottom: 2px solid #ecf0f1;
        }

        .attempt-section:last-child {
            border-bottom: none;
        }

        .attempt-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }

        .attempt-header h3 {
            color: #34495e;
            font-size: 1.3em;
        }

        .timestamp {
            color: #7f8c8d;
            font-size: 0.85em;
            margin-left: auto;
        }

        .validation-badge {
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
        }

        .validation-pass {
            background: #e8f5e9;
            color: #2e7d32;
        }

        .validation-fail {
            background: #ffebee;
            color: #c62828;
        }

        .action-comparison {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }

        .action-item {
            margin: 10px 0;
        }

        .action-item strong {
            display: block;
            margin-bottom: 5px;
            color: #555;
        }

        .action-item code {
            display: block;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        .action-correct {
            background: #e8f5e9;
            color: #2e7d32;
            border: 2px solid #4caf50;
        }

        .action-wrong {
            background: #ffebee;
            color: #c62828;
            border: 2px solid #f44336;
        }

        .gpt4o-feedback {
            margin-bottom: 20px;
            border: 2px solid #9c27b0;
            border-radius: 6px;
            overflow: hidden;
        }

        .gpt4o-feedback .section-title {
            background: #9c27b0;
            color: white;
        }

        .messages-container,
        .message-section,
        .response-section,
        .gpt4o-feedback {
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
        }

        .section-title {
            background: #34495e;
            color: white;
            padding: 12px 15px;
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            font-weight: 600;
        }

        .section-title:hover {
            background: #2c3e50;
        }

        .toggle-icon {
            margin-right: 10px;
            transition: transform 0.3s;
            display: inline-block;
        }

        .collapsed .toggle-icon {
            transform: rotate(-90deg);
        }

        .section-content {
            padding: 15px;
            background: white;
        }

        .collapsed .section-content {
            display: none;
        }

        .role-system .section-title {
            background: #3498db;
        }

        .role-examples .section-title {
            background: #16a085;
        }

        .role-query .section-title {
            background: #e67e22;
        }

        .role-assistant .section-title {
            background: #9b59b6;
        }

        pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .multimodal-content {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .image-container {
            margin: 10px 0;
        }

        .screenshot-thumbnail {
            max-width: 600px;
            width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s, border-color 0.2s;
        }

        .screenshot-thumbnail:hover {
            transform: scale(1.02);
            border-color: #3498db;
        }

        .image-modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            cursor: pointer;
        }

        .image-modal.active {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .modal-image {
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
        }

        .close-modal {
            position: absolute;
            top: 20px;
            right: 40px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }

        .close-modal:hover {
            color: #fff;
        }
        """

    def _build_javascript(self) -> str:
        """Build JavaScript for interactivity"""
        return """
        function toggleSection(element) {
            const parent = element.parentElement;
            parent.classList.toggle('collapsed');
        }

        function openImageModal(imageId) {
            const modal = document.getElementById('modal-' + imageId);
            if (modal) {
                modal.classList.add('active');
            }
        }

        function closeImageModal(imageId) {
            const modal = document.getElementById('modal-' + imageId);
            if (modal) {
                modal.classList.remove('active');
            }
        }

        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                const modals = document.querySelectorAll('.image-modal.active');
                modals.forEach(modal => modal.classList.remove('active'));
            }
        });
        """


def main():
    """Test function"""
    print("=== PromptVisualizer Module ===")
    print("This module generates HTML visualizations of prompts and responses.")
    print("Use PromptVisualizer.generate_html() to create visualizations.")


if __name__ == "__main__":
    main()
