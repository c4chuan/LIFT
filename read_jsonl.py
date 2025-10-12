#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool script for reading and analyzing sup_rollout_data_dir/1.jsonl file

This file contains training data for web navigation tasks. Each line is a separate JSON object,
containing input (system prompt and user task), output (model response), score, step, and images.
"""

import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Read jsonl file where each line is a JSON object

    Args:
        file_path: path to jsonl file

    Returns:
        list containing all JSON objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    return data


def parse_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse single entry and extract key information

    Args:
        entry: single JSON object

    Returns:
        parsed structured data
    """
    parsed = {
        'input': entry.get('input', ''),
        'output': entry.get('output', ''),
        'score': entry.get('score', 0.0),
        'step': entry.get('step', 0),
        'has_images': len(entry.get('images', [])) > 0,
        'num_images': len(entry.get('images', [])),
    }

    # Extract key information from input
    input_text = entry.get('input', '')
    if 'OBJECTIVE:' in input_text:
        # Extract task objective
        objective_start = input_text.find('OBJECTIVE:') + len('OBJECTIVE:')
        objective_end = input_text.find('\n', objective_start)
        if objective_end == -1:
            objective_end = len(input_text)
        parsed['objective'] = input_text[objective_start:objective_end].strip()

    if 'URL:' in input_text:
        # Extract URL
        url_start = input_text.find('URL:') + len('URL:')
        url_end = input_text.find('\n', url_start)
        if url_end == -1:
            url_end = len(input_text)
        parsed['url'] = input_text[url_start:url_end].strip()

    # Extract action from output
    output_text = entry.get('output', '')
    if '<action>' in output_text and '</action>' in output_text:
        action_start = output_text.find('<action>') + len('<action>')
        action_end = output_text.find('</action>', action_start)
        parsed['action'] = output_text[action_start:action_end].strip()

    return parsed


def save_images(entry: Dict[str, Any], output_dir: str, entry_idx: int) -> List[str]:
    """
    Save base64 encoded images to files

    Args:
        entry: entry containing image data
        output_dir: output directory
        entry_idx: entry index

    Returns:
        list of saved image file paths
    """
    images = entry.get('images', [])
    saved_paths = []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_idx, img_data in enumerate(images):
        if img_data.startswith('iVBOR'):  # PNG base64 data
            # Decode base64
            img_bytes = base64.b64decode(img_data)

            # Save image
            img_filename = f"entry_{entry_idx}_step_{entry.get('step', 0)}_img_{img_idx}.png"
            img_path = output_path / img_filename

            with open(img_path, 'wb') as f:
                f.write(img_bytes)

            saved_paths.append(str(img_path))

    return saved_paths


def analyze_data(data: List[Dict[str, Any]]) -> None:
    """
    Analyze dataset statistics

    Args:
        data: data list
    """
    print(f"\n=== Dataset Statistics ===")
    print(f"Total entries: {len(data)}")

    # Step distribution
    steps = [entry.get('step', 0) for entry in data]
    print(f"Step range: {min(steps)} - {max(steps)}")

    # Score distribution
    scores = [entry.get('score', 0.0) for entry in data]
    print(f"Score range: {min(scores):.2f} - {max(scores):.2f}")
    print(f"Average score: {sum(scores)/len(scores):.2f}")

    # Image count
    total_images = sum(len(entry.get('images', [])) for entry in data)
    print(f"Total images: {total_images}")
    print(f"Average images per entry: {total_images/len(data):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Read and analyze JSONL format training data')
    parser.add_argument('--file', '-f', type=str,
                       default='sup_rollout_data_dir/600.jsonl',
                       help='JSONL file path')
    parser.add_argument('--parse', '-p', action='store_true',
                       help='Parse and display structured information for each entry')
    parser.add_argument('--save-images', '-s', type=str,
                       help='Save images to specified directory')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Display dataset statistics')
    parser.add_argument('--limit', '-l', type=int,
                       help='Only process first N entries')

    args = parser.parse_args()

    # Read data
    print(f"Reading file: {args.file}")
    data = read_jsonl(args.file)
    print(f"Successfully read {len(data)} entries")

    # Limit data
    if args.limit:
        data = data[:args.limit]
        print(f"Limited to first {args.limit} entries")

    # Analyze data
    if args.analyze:
        analyze_data(data)

    # Parse data
    if args.parse:
        print("\n=== Data Parsing ===")
        for idx, entry in enumerate(data):
            parsed = parse_entry(entry)
            print(f"\n--- Entry {idx + 1} ---")
            print(f"Step: {parsed['step']}")
            print(f"Score: {parsed['score']}")
            if 'url' in parsed:
                print(f"URL: {parsed['url']}")
            if 'objective' in parsed:
                print(f"Objective: {parsed['objective']}")
            if 'action' in parsed:
                print(f"Action: {parsed['action']}")
            print(f"Number of images: {parsed['num_images']}")

    # Save images
    if args.save_images:
        print(f"\n=== Saving images to {args.save_images} ===")
        total_saved = 0
        for idx, entry in enumerate(data):
            saved = save_images(entry, args.save_images, idx)
            total_saved += len(saved)
            if saved:
                print(f"Entry {idx + 1}: Saved {len(saved)} images")
        print(f"Total saved {total_saved} images")


if __name__ == '__main__':
    main()
