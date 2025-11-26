"""
Prompt 构造模块

负责构建用于生成推理过程的 prompt
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image


# Examples for reasoning generation (adapted from prompts.py)
REASONING_EXAMPLES = {
    "LIFT": [
        {
            "query": """URL: http://classifieds.com/index.php?page=search&sCategory=17
OBJECTIVE: Explore the "Furniture" category of Washington, D.C. and find me the most recent blue chair.
PREVIOUS ACTIONS: click [40]
TARGET ACTION: type [7] [Washington] [0]

Please generate detailed reasoning that explains why the target action should be performed.""",
            "reasoning": """Let's observe step by step. First, I will zoom in to observe the overall page structure.

<zoom in>
Looking at the page layout, I can see it's divided into two main areas: a Search Filters Area on the left side and a Main Content Area on the right. The page header shows "Furniture" category, indicating we're already browsing furniture items. The main content displays several furniture listings including sofas, chairs, and mattresses with their titles and preview images.
</zoom in>

<shift>
Now let me examine the Search Filters Area more carefully. I can see several filter options arranged vertically: "Your search" section at the top, followed by a "City" text input field, a "Show only listings with pictures" checkbox, and "Price Min./Max." input fields below.
</shift>

The objective requires finding items in Washington, D.C., so I need to use the city filter. Let me zoom in on the City input field.

<zoom in>
Focusing on the "City" input field in the Search Filters Area, I observe its bounding box with a distinct border color. By carefully examining the border, I can identify this element has a specific color-coded border. Looking at the numerical id that shares this same border color, I can determine this is element [7].
</zoom in>

<shift>
Let me also check the current listings shown in the Main Content Area. I see items from various locations, confirming that the results are not yet filtered by city. The listings include furniture from different areas, which means applying the city filter will help narrow down to Washington, D.C. specific items.
</shift>

<summary>
Observations so far:
1. The page displays furniture items from various locations in the Main Content Area
2. The Search Filters Area on the left contains a "City" input field
3. By examining the border color, I identified the City field as element [7]
4. To find the most recent blue chair in Washington, D.C., I must first filter by city
5. The target action is to type "Washington" in field [7] without pressing Enter (flag [0]), which will filter the results

So the next action I will perform is type [7] [Washington] [0]
</summary>

<action>
type [7] [Washington] [0]
</action>
""",
            "image_path": "data/example/example.png"
        }
    ],
    "ORIGINAL": [
        {
            "query": """URL: http://onestopmarket.com/office-products/office-electronics.html
OBJECTIVE: What is the price of HP Inkjet Fax Machine?
PREVIOUS ACTIONS: None
TARGET ACTION: stop [$279.49]

Please generate detailed reasoning that explains why the target action should be performed.""",
            "reasoning": """Let's think step-by-step. This page lists the information of HP Inkjet Fax Machine, which is the product identified in the objective. Looking at the page, I can see:
- [1744] link 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'
- [1749] StaticText '$279.49'
- [1757] button 'Add to Cart'

The price of the HP Inkjet Fax Machine is clearly displayed as $279.49. I have found the information requested in the objective. Therefore, I should issue the stop action with the answer.

In summary, the next action I will perform is ```stop [$279.49]```""",
            "image_path": "visualwebarena/agent/prompts/som_examples/som_example1.png"
        }
    ]
}

URL_MAPPINGS = {'http://127.0.0.1:4399': 'http://homepage.com', 'http://127.0.0.1:7770': 'http://onestopmarket.com', 'http://127.0.0.1:8888': 'http://wikipedia.org', 'http://127.0.0.1:9980': 'http://classifieds.com', 'http://127.0.0.1:9999': 'http://reddit.com'}
def map_url_to_real(url: str) -> str:
    """Map the urls to their real world counterparts"""
    for i, j in URL_MAPPINGS.items():
        if i in url:
            url = url.replace(i, j)
    return url

class PromptBuilder:
    """Prompt 构建器"""

    def __init__(self, prompt_style: str = "LIFT", use_examples: bool = True):
        """
        初始化 Prompt 构建器

        Args:
            prompt_style: Prompt 风格，"LIFT" 或 "ORIGINAL"
            use_examples: 是否使用例子
        """
        self.prompt_style = prompt_style
        self.use_examples = use_examples
        self.system_prompt = self._get_system_prompt()

        # Load examples
        self.examples = REASONING_EXAMPLES.get(prompt_style, []) if use_examples else []

    def _get_system_prompt(self) -> str:
        """Get system prompt"""
        if self.prompt_style == "LIFT":
            return """You are an autonomous intelligent agent tasked with navigating a web browser. Your task is to generate detailed reasoning that explains why a given target action should be performed, based on the current page state, previous actions, and the objective.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page screenshot: This is a screenshot of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.
The current web page's URL: This is the page you're currently navigating.
The previous actions: These are the actions that have been performed. It may be helpful to track the progress.
The target action: This is the ground truth action that should be performed next.

The actions can fall into several categories:

Page Operation Actions:
<action>click [id]</action>: This action clicks on an element with a specific id on the webpage.
<action>type [id] [content]</action>: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0, i.e., ```type [id] [content] [0]```.
<action>hover [id]</action>: Hover over an element with id.
<action>press [key_comb]</action>: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
<action>scroll [down]</action> or <action>scroll [up]</action>: Scroll the page up or down.

Tab Management Actions:
<action>new_tab</action>: Open a new, empty browser tab.
<action>tab_focus [tab_index]</action>: Switch the browser's focus to a specific tab using its index.
<action>close_tab</action>: Close the currently active tab.

URL Navigation Actions:
<action>goto [url]</action>: Navigate to a specific URL.
<action>go_back</action>: Navigate to the previously viewed page.
<action>go_forward</action>: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
<action>stop [answer]</action>: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket.

Perceive the environment and generate reasoning:
You should take two observing strategies, zoom in and shift, to perceive the environment, then aggregate the observation and form the reasoning for the target action.
<zoom in></zoom in>: take a closer look at the details in the screenshot. The area you observe in this strategy should focus on relevant elements for the target action.
<shift></shift>: focus on other informative areas in the screenshot that haven't been explored in your previous observations.
<summary></summary>: summarize all observation results and explain why the target action is reasonable and necessary.

To be successful, it is very important to follow the following rules:
1. Your reasoning should be detailed and explain how you observe the page to reach the target action.
2. Your reasoning should be logically clear with explicit steps.
3. The content of each observation must be contained within <zoom in></zoom in> and <shift></shift>, your summary in <summary></summary> and your action should be contained within <action></action>..
4. Your summary MUST explain why the target action is the correct next step.
5. Do NOT use phase markers like "Phase 1", "Phase 2", "phase-1", "phase-2" or similar numbered phase labels in your reasoning. Keep your reasoning natural and flowing without artificial phase divisions.
"""
        else:
            return """You are an autonomous intelligent agent tasked with navigating a web browser. Your task is to generate detailed reasoning that explains why a given target action should be performed, based on the current page state, previous actions, and the objective.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page screenshot: This is a screenshot of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.
The observation, which lists the IDs of all interactable elements on the current web page with their text content if any, in the format [id] [tagType] [text content]. tagType is the type of the element, such as button, link, or textbox. text content is the text content of the element. For example, [1234] [button] ['Add to Cart'] means that there is a button with id 1234 and text content 'Add to Cart' on the current web page. [] [StaticText] [text] means that the element is of some text that is not interactable.
The current web page's URL: This is the page you're currently navigating.
The previous actions: These are the actions that have been performed. It may be helpful to track the progress.
The target action: This is the ground truth action that should be performed next.

The actions can fall into several categories:

Page Operation Actions:
```click [id]```: This action clicks on an element with a specific id on the webpage.
```type [id] [content]```: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0, i.e., ```type [id] [content] [0]```.
```hover [id]```: Hover over an element with id.
```press [key_comb]```: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
```scroll [down]``` or ```scroll [up]```: Scroll the page up or down.

Tab Management Actions:
```new_tab```: Open a new, empty browser tab.
```tab_focus [tab_index]```: Switch the browser's focus to a specific tab using its index.
```close_tab```: Close the currently active tab.

URL Navigation Actions:
```goto [url]```: Navigate to a specific URL.
```go_back```: Navigate to the previously viewed page.
```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
```stop [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket.

To be successful, it is very important to follow the following rules:
1. Your reasoning should analyze the key information on the current page.
2. Your reasoning should consider the previous actions and the task objective.
3. Your reasoning should explain why the target action is appropriate and necessary.
4. Generate the reasoning in the correct format. Start with reasoning step by step, and end with a "In summary, the next action I will perform is" phrase, followed by the target action inside ``````. For example, "In summary, the next action I will perform is ```click [1234]```".
5. Do NOT use phase markers like "Phase 1", "Phase 2", "phase-1", "phase-2" or similar numbered phase labels in your reasoning.
"""

    def build_reasoning_prompt(
        self,
        intent: str,
        current_url: str,
        current_screenshot: Image.Image,
        previous_actions: List[str],
        ground_truth_action: str,
        input_images: Optional[List[Image.Image]] = None,
        history_reasonings: Optional[List[Dict[str, Any]]] = None,
        current_observation_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Build reasoning prompt in multi-turn conversation format

        Args:
            intent: Task objective
            current_url: Current page URL
            current_screenshot: Current page screenshot
            previous_actions: List of previous actions
            ground_truth_action: Ground truth action
            input_images: Input image list (task-related images)
            history_reasonings: List of previous reasoning pairs in this trajectory
                Format: [{"screenshot": Image, "action": str, "reasoning": str}, ...]
            current_observation_text: Current page observation text (element list with IDs)

        Returns:
            Message list in Qwen VL format
        """
        if input_images is None:
            input_images = []
        if history_reasonings is None:
            history_reasonings = []

        # Build previous actions string
        if previous_actions:
            history_str = "\n".join([f"- {action}" for action in previous_actions[-5:]])  # Keep only last 5 actions
        else:
            history_str = "None"

        # Build current query text
        current_query = f"""URL: {map_url_to_real(current_url)}
OBJECTIVE: {intent}
PREVIOUS ACTIONS:
{history_str}
TARGET ACTION: {map_url_to_real(ground_truth_action)}"""

        # 仅为 LIFT 风格添加 OBSERVATION（放在 TARGET ACTION 之后）
        if self.prompt_style == "LIFT" and current_observation_text:
            current_query += f"""
OBSERVATION:
{current_observation_text}"""

        current_query += """

Please generate detailed reasoning that explains why the target action should be performed.

IMPORTANT INSTRUCTIONS:
1. You MUST perform at least 5 observation steps following this structured approach:

   Phase 1 - Global Layout (REQUIRED):
   Use <zoom in> to observe the overall page layout and identify all major functional areas (navigation bar, sidebar, main content area, footer, etc.)

   Phase 2 - Area Exploration (REQUIRED):
   Use <shift> to explore different areas of the page, including information that seems both related and unrelated to the task. Check the distribution and organization of page elements.

   Phase 3 - Element Comparison (REQUIRED):
   Use <zoom in> to observe multiple candidate elements that could potentially be relevant. Compare and analyze their characteristics. Explain why certain elements are NOT suitable choices.

   Phase 4 - Target Focus (REQUIRED):
   Use <zoom in> to focus on the target element. Carefully observe its BORDER COLOR to determine the som_id. Check its attributes and state.

   Phase 5 - Decision Justification (REQUIRED):
   Summarize why this specific element is the best choice to accomplish the task, rather than other elements on the page.

2. Your reasoning should demonstrate comprehensive page understanding, not just focus on the target element.
3. When identifying web elements, use <zoom in> to carefully observe BORDER COLOR to determine som_id.
4. Follow this format strictly: Start with "Let's observe step by step", use multiple <zoom in></zoom in> and <shift></shift> tags (at least 5 observations total) with detailed content, then provide <summary></summary> that ends with "So the next action I will perform is [action]", and finally wrap the action in <action></action> tags."""

        # Start building messages with system prompt
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        # Add examples (if enabled)
        if self.use_examples and self.examples:
            for example in self.examples:
                # Load example image
                example_image = self._load_example_image(example["image_path"])
                if example_image is not None:
                    # Add example user message
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": example["query"]},
                            {"type": "text", "text": "\nCurrent page screenshot:"},
                            {"type": "image", "image": example_image}
                        ]
                    })
                    # Add example assistant message
                    messages.append({
                        "role": "assistant",
                        "content": example["reasoning"]
                    })

        # Add history reasoning pairs (if provided)
        for hist in history_reasonings:
            hist_screenshot = hist.get("screenshot")
            hist_action = hist.get("action", "")
            hist_reasoning = hist.get("reasoning", "")

            if hist_screenshot is not None and hist_reasoning:
                hist_query = f"""URL: {current_url}
OBJECTIVE: {intent}
PREVIOUS ACTIONS: {hist.get("previous_actions", "None")}
TARGET ACTION: {hist_action}

Please generate detailed reasoning that explains why the target action should be performed.

IMPORTANT INSTRUCTIONS:
1. You MUST perform at least 5 observation steps following this structured approach:

   Phase 1 - Global Layout (REQUIRED):
   Use <zoom in> to observe the overall page layout and identify all major functional areas (navigation bar, sidebar, main content area, footer, etc.)

   Phase 2 - Area Exploration (REQUIRED):
   Use <shift> to explore different areas of the page, including information that seems both related and unrelated to the task. Check the distribution and organization of page elements.

   Phase 3 - Element Comparison (REQUIRED):
   Observe multiple candidate elements that could potentially be relevant. Compare and analyze their characteristics. Explain why certain elements are NOT suitable choices.

   Phase 4 - Target Focus (REQUIRED):
   Use <zoom in> to focus on the target element. Carefully observe its BORDER COLOR to determine the som_id. Check its attributes and state.

   Phase 5 - Decision Justification (REQUIRED):
   Summarize why this specific element is the best choice to accomplish the task, rather than other elements on the page.

2. Your reasoning should demonstrate comprehensive page understanding, not just focus on the target element.
3. When identifying web elements, use <zoom in> to carefully observe BORDER COLOR to determine som_id.
4. Follow this format strictly: Start with "Let's observe step by step", use multiple <zoom in></zoom in> and <shift></shift> tags (at least 5 observations total) with detailed content, then provide <summary></summary> that ends with "So the next action I will perform is [action]", and finally wrap the action in <action></action> tags."""

                # Add history user message
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": hist_query},
                        {"type": "text", "text": "\nCurrent page screenshot:"},
                        {"type": "image", "image": hist_screenshot}
                    ]
                })
                # Add history assistant message
                messages.append({
                    "role": "assistant",
                    "content": hist_reasoning
                })

        # Add current query
        current_content = [
            {"type": "text", "text": current_query},
            {"type": "text", "text": "\nCurrent page screenshot:"},
            {"type": "image", "image": current_screenshot}
        ]

        # Add extra input images if provided
        if input_images:
            for idx, img in enumerate(input_images):
                current_content.extend([
                    {"type": "text", "text": f"\nInput image {idx + 1}:"},
                    {"type": "image", "image": img}
                ])

        messages.append({
            "role": "user",
            "content": current_content
        })

        return messages

    def build_simplified_prompt(
        self,
        intent: str,
        ground_truth_action: str,
        previous_action: str = "None"
    ) -> List[Dict[str, Any]]:
        """
        Build simplified prompt (without screenshot, for testing)

        Args:
            intent: Task objective
            ground_truth_action: Ground truth action
            previous_action: Previous action

        Returns:
            Message list
        """
        user_text = f"""OBJECTIVE: {intent}
PREVIOUS ACTION: {previous_action}
TARGET ACTION: {ground_truth_action}

Please generate the reasoning process from observation to executing the target action."""

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text}
        ]

    def build_correction_prompt(
        self,
        original_messages: List[Dict[str, Any]],
        generated_action: Optional[str],
        ground_truth_action: str,
        retry_count: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Build correction prompt when validation fails

        当生成的action与ground truth不匹配时，构建纠正提示，
        直接告知模型正确的action，要求其重新生成reasoning

        Args:
            original_messages: 原始的消息列表（包含system、examples、history、current query）
            generated_action: 模型生成的错误action（可能为None）
            ground_truth_action: 正确的ground truth action
            retry_count: 当前是第几次重试（1或2）

        Returns:
            新的消息列表，包含纠正提示
        """
        # 构建纠正提示
        if retry_count == 1:
            # 第1次重试：温和提示
            correction_text = f"""
Your previous reasoning generated the action: {generated_action if generated_action else "[NO ACTION TAG FOUND]"}

However, the CORRECT action should be: {ground_truth_action}

Please regenerate a detailed reasoning process that:
1. Still follows the 5-phase comprehensive observation structure (Phase 1-5 as instructed)
2. Explores the page thoroughly including different areas to demonstrate full understanding
3. Naturally leads to the conclusion that the action "{ground_truth_action}" is the correct choice
4. Identifies the specific information and element characteristics that justify this action
5. Your final <action> tag must contain EXACTLY: {ground_truth_action}

Remember: Be thorough in your observations across all phases, but ensure your conclusion matches the correct action.
"""
        else:
            # 第2次重试：加强语气，这是最后一次机会
            correction_text = f"""
IMPORTANT: You have attempted {retry_count + 1} times, but the action was still incorrect.

The CORRECT action is definitively: {ground_truth_action}

This is your LAST attempt. Please provide one final reasoning that:
1. Maintains the comprehensive 5-phase observation structure (this is REQUIRED for training quality)
2. Thoroughly explores all page areas to demonstrate deep understanding of the page
3. Provides clear justification for why "{ground_truth_action}" is the right choice
4. MUST conclude with the <action> tag containing EXACTLY: {ground_truth_action}

Critical: The <action></action> tag MUST contain precisely: {ground_truth_action}
No variations, no alternatives - it must match exactly.
"""

        # 添加纠正提示到消息列表
        correction_messages = original_messages.copy()
        correction_messages.append({
            "role": "user",
            "content": correction_text
        })

        return correction_messages

    def build_feedback_based_correction_prompt(
        self,
        original_messages: List[Dict[str, Any]],
        gpt4o_feedback: str,
        ground_truth_action: str,
        retry_count: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Build correction prompt based on GPT-4o feedback (always includes correct answer)

        当生成的action与ground truth不匹配时，基于GPT-4o的错误分析反馈构建纠正提示。
        每次重试都会同时提供GPT-4o的错误分析和正确答案。

        Args:
            original_messages: 原始的消息列表（包含system、examples、history、current query）
            gpt4o_feedback: GPT-4o 提供的错误分析反馈（英文）
            ground_truth_action: 正确的ground truth action（每次都提供）
            retry_count: 当前是第几次重试（1或2）

        Returns:
            新的消息列表，包含GPT-4o反馈和正确答案
        """
        # 构建纠正提示（全英文）
        if retry_count == 1:
            # 第1次重试：包含GPT-4o反馈 + 正确答案
            correction_text = f"""## Error Analysis from Expert Tutor

An expert tutor has analyzed your previous reasoning and identified the following issues:

{gpt4o_feedback}

## Correct Action

The correct action for this step is: **{ground_truth_action}**

## Task

Please regenerate your detailed reasoning process:

1. **Maintain the 5-phase comprehensive observation structure** (Phase 1-5 as instructed) - this is REQUIRED for training quality
2. **Address the issues** pointed out in the expert's feedback
3. **Re-observe the page** based on the expert's guidance
4. **Naturally arrive at** the conclusion that the action "{ground_truth_action}" is the correct choice
5. **Provide clear justification** for why this specific action is appropriate
6. **Your final <action> tag MUST contain EXACTLY**: {ground_truth_action}

**Important:**
- Your reasoning should demonstrate that you've understood the feedback and corrected your observation process
- Do NOT simply state "because the expert told me" - show your reasoning process
- The reasoning must appear as if you independently arrived at the correct conclusion through careful observation
"""
        else:
            # 第2次重试：更强调关键点 + GPT-4o反馈 + 正确答案
            correction_text = f"""## CRITICAL: Final Attempt - Expert Analysis

You have now attempted this task {retry_count + 1} times. This is your LAST opportunity.

An expert tutor has provided this CRITICAL analysis:

{gpt4o_feedback}

## CORRECT ACTION (FINAL ANSWER)

The correct action is DEFINITIVELY: **{ground_truth_action}**

## FINAL TASK

This is your last chance. Please provide one final reasoning that:

1. **MUST maintain the comprehensive 5-phase observation structure** (Phase 1-5) - this is ABSOLUTELY REQUIRED
2. **Carefully addresses EVERY point** raised in the expert's critical analysis above
3. **Demonstrates clear, step-by-step observation** following the expert's guidance
4. **Shows understanding of why** "{ground_truth_action}" is the correct and ONLY appropriate action
5. **Provides detailed justification** for the correct element choice
6. **MUST conclude with <action> tag containing EXACTLY**: {ground_truth_action}

**CRITICAL REMINDERS:**
- This is your final attempt - accuracy is paramount
- Follow the expert's step-by-step guidance precisely
- Show your complete observation and reasoning process
- Do NOT just state the answer - demonstrate HOW you arrived at it through observation
- The <action></action> tag MUST contain precisely: {ground_truth_action}
- No variations, no alternatives - exact match required
"""

        # 添加纠正提示到消息列表
        correction_messages = original_messages.copy()
        correction_messages.append({
            "role": "user",
            "content": correction_text
        })

        return correction_messages

    @staticmethod
    def extract_screenshot_from_state(state_info: Dict[str, Any]) -> Image.Image:
        """
        Extract screenshot from StateInfo

        Args:
            state_info: StateInfo dictionary

        Returns:
            PIL Image object
        """
        try:
            # StateInfo structure: {"observation": {...}, "info": {...}}
            # observation may contain "image", "image_som", etc.
            observation = state_info.get("observation", {})

            # Try different key names
            for key in ["image", "image_som", "screenshot"]:
                if key in observation:
                    img_data = observation[key]

                    # If it's a numpy array
                    if hasattr(img_data, "shape"):
                        return Image.fromarray(img_data)

                    # If it's a PIL Image
                    if isinstance(img_data, Image.Image):
                        return img_data

                    # If it's bytes
                    if isinstance(img_data, bytes):
                        return Image.open(BytesIO(img_data))

            raise ValueError("No valid screenshot data found in StateInfo")

        except Exception as e:
            raise Exception(f"Failed to extract screenshot from StateInfo: {e}")

    @staticmethod
    def extract_url_from_state(state_info: Dict[str, Any]) -> str:
        """
        Extract URL from StateInfo

        Args:
            state_info: StateInfo dictionary

        Returns:
            URL string
        """
        try:
            info = state_info.get("info", {})
            page = info.get("page")
            if page and hasattr(page, "url"):
                return map_url_to_real(page.url)
            return info.get("url", "unknown")
        except Exception as e:
            print(f"Failed to extract URL: {e}")
            return "unknown"

    @staticmethod
    def extract_observation_text_from_state(state_info: Dict[str, Any]) -> str:
        """
        从 StateInfo 中提取 observation text

        Args:
            state_info: StateInfo 字典

        Returns:
            观察文本字符串，如果提取失败则返回空字符串
        """
        try:
            observation = state_info.get("observation", {})
            text = observation.get("text", "")
            return text if text else ""
        except Exception as e:
            print(f"警告: 提取 observation text 失败: {e}")
            return ""

    @staticmethod
    def _load_example_image(image_path: str) -> Optional[Image.Image]:
        """
        Load example image from path

        Args:
            image_path: Path to the example image (relative to project root)

        Returns:
            PIL Image object or None if loading fails
        """
        try:
            # Try to load from absolute path first
            if Path(image_path).is_absolute() and Path(image_path).exists():
                return Image.open(image_path)

            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            full_path = project_root / image_path

            if full_path.exists():
                return Image.open(full_path)

            # Try without project root (direct path)
            if Path(image_path).exists():
                return Image.open(image_path)

            print(f"Warning: Example image not found: {image_path}")
            return None

        except Exception as e:
            print(f"Warning: Failed to load example image {image_path}: {e}")
            return None


def main():
    """Test function"""
    builder = PromptBuilder(prompt_style="LIFT")

    # Test simplified prompt
    print("=== Test Simplified Prompt ===")
    messages = builder.build_simplified_prompt(
        intent='Search for "laptop" and find the cheapest one',
        ground_truth_action='click [123]',
        previous_action='type [456] [laptop] [1]'
    )

    print("System prompt:")
    print(messages[0]["content"])
    print("\nUser message:")
    print(messages[1]["content"])


if __name__ == "__main__":
    main()
