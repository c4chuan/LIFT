"""
GPT-4o Feedback Generator Module

Provides intelligent error analysis for reasoning correction using GPT-4o.
"""

import os
import base64
from io import BytesIO
from typing import Optional
from PIL import Image
from openai import OpenAI


class GPT4oFeedbackGenerator:
    """Use GPT-4o to generate intelligent feedback for reasoning errors"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "gpt-4o",
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        Initialize GPT-4o client

        Args:
            api_key: OpenAI API key
            base_url: OpenAI base URL
            model: Model name (default: gpt-4o)
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout
        )
        self.model = model

    def analyze_error(
        self,
        intent: str,
        current_url: str,
        observation_text: str,
        screenshot: Image.Image,
        generated_reasoning: str,
        generated_action: str,
        ground_truth_action: str,
        retry_count: int = 1
    ) -> str:
        """
        Analyze the model's error and generate feedback

        Args:
            intent: Task objective
            current_url: Current page URL
            observation_text: Page element text information
            screenshot: Page screenshot
            generated_reasoning: Model's generated reasoning
            generated_action: Model's generated incorrect action
            ground_truth_action: Correct action
            retry_count: Which retry attempt (1 or 2)

        Returns:
            GPT-4o's error analysis feedback text (in English)
        """
        # Convert screenshot to base64
        screenshot_b64 = self._image_to_base64(screenshot)

        # Build prompt based on retry count
        if retry_count == 1:
            prompt = self._build_first_retry_prompt(
                intent, current_url, observation_text,
                generated_reasoning, generated_action, ground_truth_action
            )
        else:  # retry_count == 2
            prompt = self._build_second_retry_prompt(
                intent, current_url, observation_text,
                generated_reasoning, generated_action, ground_truth_action
            )

        # Call GPT-4o API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_b64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )

            feedback = response.choices[0].message.content
            return feedback

        except Exception as e:
            print(f"Warning: GPT-4o API call failed: {e}")
            # Fallback to simple feedback
            return f"Your generated action '{generated_action}' is incorrect. The correct action should be '{ground_truth_action}'. Please re-examine the page carefully."

    def _build_first_retry_prompt(
        self,
        intent: str,
        url: str,
        observation: str,
        reasoning: str,
        wrong_action: str,
        correct_action: str
    ) -> str:
        """Build prompt for first retry (detailed but encouraging)"""
        return f'''You are an expert tutor for web navigation tasks. An AI agent made a mistake while performing a web task, and you need to help it understand what went wrong.

## Task Information
**Objective:** {intent}
**Current URL:** {url}

## Page Information
**Screenshot:** (See attached image)

**Page Elements (OBSERVATION):**
{observation}

## Agent's Output
**Agent's Complete Reasoning:**
{reasoning}

**Agent's Generated Action:** {wrong_action}

## Error Analysis Task
The agent's action is incorrect. The correct action should be: **{correct_action}**

Please provide a detailed error analysis covering these aspects:

1. **Missing Observations**: In the Phase 1-5 observations, what key information did the agent miss? Which page areas or elements were not sufficiently examined?

2. **Understanding Errors**: Did the agent misunderstand any page elements, the task objective, or the context?

3. **Logic Issues**: Where did the reasoning process from observation to action have logical gaps or incorrect inferences?

4. **Element Confusion**: Did the agent confuse different elements? If so, point out the distinguishing features between the confused elements.

5. **Correct Element Characteristics**: What are the specific characteristics of the correct element (ID: check border color in screenshot, position, text content, visual appearance)?

6. **Action Guidance**: Provide clear guidance on what observation strategy the agent should use and what key points to focus on to arrive at the correct action.

**Important:**
- Provide specific, actionable feedback
- Explain WHY the correct action is appropriate in the context of the task objective
- Help the agent understand the reasoning path from observation to the correct action
- Keep your feedback constructive and educational'''

    def _build_second_retry_prompt(
        self,
        intent: str,
        url: str,
        observation: str,
        reasoning: str,
        wrong_action: str,
        correct_action: str
    ) -> str:
        """Build prompt for second retry (more direct and emphatic)"""
        return f'''You are an expert tutor for web navigation tasks. An AI agent has attempted twice but still made mistakes. This is the FINAL opportunity, so you need to provide VERY CLEAR and DIRECT guidance.

## Task Information
**Objective:** {intent}
**Current URL:** {url}

## Page Information
**Screenshot:** (See attached image)

**Page Elements (OBSERVATION):**
{observation}

## Agent's Latest Output
**Agent's Complete Reasoning:**
{reasoning}

**Agent's Generated Action:** {wrong_action}

## Critical Analysis Task
The agent has failed twice. The correct action MUST be: **{correct_action}**

This is the LAST attempt. Provide an emphatic and crystal-clear analysis:

1. **Critical Errors**: What are the MOST CRITICAL mistakes the agent is making repeatedly?

2. **Correct Element - Exact Identification**:
   - What is the EXACT element (with ID) needed for the correct action?
   - How to identify it by border color in the screenshot?
   - What is its exact position on the page?
   - What is its exact text content or label?

3. **Why This Element**: Explain clearly WHY this specific element is the only correct choice given the task objective.

4. **Contrast with Wrong Choice**: If the agent chose a wrong element, explain the KEY DIFFERENCES between the wrong element and the correct one.

5. **Step-by-Step Guidance**: Provide a CLEAR, STEP-BY-STEP observation path:
   - First, observe...
   - Then, identify...
   - Finally, conclude...

6. **Absolute Clarity**: State unambiguously what the agent MUST do to complete this task correctly.

**CRITICAL:**
- This is the final chance - be as direct and clear as possible
- Use emphatic language to stress the key points
- Ensure the agent cannot misunderstand your guidance
- Make the reasoning path from observation to correct action crystal clear'''

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """
        Convert PIL Image to base64 PNG string (lossless, no resize)

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded PNG string
        """
        # Convert to RGB if RGBA (for PNG compatibility)
        if image.mode == 'RGBA':
            # Create white background for transparency
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        elif image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        # Save to bytes as PNG (lossless)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def main():
    """Test function"""
    print("=== GPT4oFeedbackGenerator Module ===")
    print("This module provides intelligent error analysis using GPT-4o.")
    print("Configure your API key and base URL in config.yaml to use it.")


if __name__ == "__main__":
    main()
