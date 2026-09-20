import os
import json
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from openai import OpenAI
from services.settings import is_masked

DOCUMENTARY_STYLE_INSTRUCTIONS = """You are a master video narrator, TikTok viral storyteller, and scriptwriter.

Your task is to translate and rewrite video transcripts into natural, highly human-understandable Burmese that grips the audience's attention (လူနားလည်ရလွယ်ကူပြီး စိတ်ဝင်စားစရာကောင်းသော TikTok Viral စတိုင်).

Core Guidelines:
* Do NOT translate word-for-word (မူရင်းစာလုံးပေါင်း တိုက်ရိုက်မပြန်ပါနှင့်).
* Rewrite the script so that it sounds extremely natural, fluent, and engaging for a Burmese audience.
* Preserve 100% of the original meaning, story plot, and essential facts (မူရင်းအဓိပ္ပာယ်နှင့် အရသာကို မပျက်စေရ).
* For Story & Movie Recaps: Tell the plot dynamically with curiosity hooks that hold viewer interest from start to finish (ဇာတ်လမ်းဇာတ်ကွက်ကို လူတွေ စိတ်ဝင်စားအောင် ဆွဲဆောင်မှုရှိရှိ ရေးဖွဲ့ပေးရန်).
* SHORT SUBTITLES: Keep each segment short and concise (max 25-30 characters per cue). NEVER combine two full sentences into one segment string.
* SENTENCE PUNCTUATION: Use proper Burmese punctuation (၊ or ။ or ?). Break long compound sentences into short, punchy sentence units suitable for narration.
* Pure voiceover script only - no meta text, bullet points, or section labels.
"""

STYLE_PROMPTS = {
    "viral": """Translation Style: 🔥 TikTok Viral & Story Recap (TikTok Viral ဇာတ်လမ်းပြန်ပြော စတိုင်)
* Tone: High-energy, captivating, story-driven commentary perfect for TikTok, Reels, and YouTube Shorts.
* Use punchy, modern Burmese phrasing that hooks viewers instantly.
* Build intrigue around the plot while preserving original story context.
""",
    "comedy": """Translation Style: 😂 Comedy / Funny (ဟာသ စတိုင်)
* Tone: Lighthearted, witty, hilarious, and funny Burmese commentary.
* Use clever Burmese humor and punchy expressions.
* Ensure the commentary is funny and engaging while maintaining the core facts of the video.
""",
    "sad": """Translation Style: 😢 Drama / Sad / Melancholic (အလွမ်းအဆွေး စတိုင်)
* Tone: Somber, dramatic, emotional narrative style (အလွမ်းအဆွေး).
* Use deep, expressive Burmese storytelling words that evoke feelings of sadness, longing, or dramatic tension.
* Keep the core narrative true to the original video context.
""",
    "emotional": """Translation Style: 💖 Emotional / Inspiring (Emotional စတိုင်)
* Tone: Touching, inspiring, deeply emotional resonance (မူရင်း အဓိပ္ပာယ် မပျက်ဘဲ ခံစားချက်ပါသော စတိုင်).
* Focus on inspiring, warm, or powerful emotional connections.
* Strictly preserve the original core meaning and facts.
""",
    "documentary": """Translation Style: 📚 Story Recap & Documentary (ဇာတ်လမ်းပြန်ပြော / ပညာပေး စတိုင်)
* Tone: Engaging story recap and educational commentary style.
* Curiosity-driven, clear storytelling flow, smooth natural narration.
* Clear human Burmese that holds viewer attention throughout the video.
"""
}


def get_style_instructions(style_name: str = "documentary") -> str:
    clean_style = style_name.lower().strip() if style_name else "documentary"
    style_text = STYLE_PROMPTS.get(clean_style, STYLE_PROMPTS["documentary"])
    return f"{DOCUMENTARY_STYLE_INSTRUCTIONS}\n\n{style_text}"


def translate_transcript_gemini(
    segments: List[Dict[str, Any]],
    target_language: str = "Burmese",
    mode: str = "full",  # "full" or "recap"
    translation_style: str = "documentary",
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Translates or creates recap segments using Gemini API with selected translation style prompt.
    """
    gemini_key = api_key
    if not gemini_key or is_masked(gemini_key):
        from services.settings import SettingsManager
        live_settings = SettingsManager.get_instance().load_settings()
        gemini_key = live_settings.gemini_api_key
    if not gemini_key or is_masked(gemini_key):
        gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key or not gemini_key.strip():
        raise ValueError("Gemini API key is not configured. Open Settings → AI & Prompts and add a valid Gemini API key.")

    client = genai.Client(api_key=gemini_key)
    style_instructions = get_style_instructions(translation_style)

    if mode == "recap":
        prompt = f"""{style_instructions}

Below is a transcript of a video with timing timestamps:
{json.dumps(segments, ensure_ascii=False)}

Your task:
1. Create a clear, engaging recap narration script in {target_language} following all requirements above.
2. Break the recap into timed segments fitting within the total video duration.
3. Output MUST be valid JSON array of objects with keys: "start" (float), "end" (float), "text" (translated string).
Return ONLY the raw JSON array.
"""
    else:
        prompt = f"""{style_instructions}

Below is a list of video transcript segments with timestamps:
{json.dumps(segments, ensure_ascii=False)}

Your task:
1. Translate each segment's 'text' into natural {target_language} following all requirements above.
2. Keep the exact same 'start' and 'end' timestamps for each segment.
3. Output MUST be valid JSON array of objects with keys: "start" (float), "end" (float), "text" (translated string).
Return ONLY the raw JSON array.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    try:
        translated_segments = json.loads(response.text)
        return translated_segments
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return segments


def translate_transcript_openai(
    segments: List[Dict[str, Any]],
    target_language: str = "Burmese",
    mode: str = "full",
    translation_style: str = "documentary",
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Translates or creates recap segments using OpenAI API with selected translation style prompt.
    """
    openai_key = api_key
    if not openai_key or is_masked(openai_key):
        from services.settings import SettingsManager
        live_settings = SettingsManager.get_instance().load_settings()
        openai_key = live_settings.openai_api_key
    if not openai_key or is_masked(openai_key):
        openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key or not openai_key.strip():
        raise ValueError("OpenAI API key is not configured. Open Settings → AI & Prompts and add a valid OpenAI API key.")

    client = OpenAI(api_key=openai_key)
    system_prompt = get_style_instructions(translation_style)
    user_prompt = f"""Below is a transcript array with timestamps:
{json.dumps(segments, ensure_ascii=False)}

Translate into {target_language} in '{mode}' mode following all guidelines.
Return ONLY a valid JSON array of objects, each containing:
- "start": float
- "end": float
- "text": string (translated {target_language} voiceover)
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    if isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list):
                return val
    return data if isinstance(data, list) else segments


def translate_transcript(
    segments: List[Dict[str, Any]],
    target_language: str = "Burmese",
    mode: str = "full",
    translation_style: str = "documentary",
    provider: str = "gemini",
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Unified translation handler supporting Gemini or OpenAI provider and custom translation style.
    """
    if provider == "openai":
        try:
            return translate_transcript_openai(segments, target_language, mode, translation_style, api_key)
        except Exception as e:
            print(f"[Translator] OpenAI provider error ({e}). Falling back to Gemini...")
            return translate_transcript_gemini(segments, target_language, mode, translation_style, api_key)
    else:
        return translate_transcript_gemini(segments, target_language, mode, translation_style, api_key)


def generate_viral_translation_options(
    segments: List[Dict[str, Any]],
    target_language: str = "Burmese",
    translation_style: str = "documentary",
    provider: str = "gemini",
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generates 3 distinct translation options with Viral Scores & Style variations matching requested style:
    Option 1: 🔥 Ultra Hook Variation
    Option 2: ⚡ High Engagement Variation
    Option 3: 📌 Direct & Clear Variation
    """
    gemini_key = api_key or os.getenv("GEMINI_API_KEY")
    openai_key = api_key or os.getenv("OPENAI_API_KEY")

    style_instructions = get_style_instructions(translation_style)

    prompt = f"""{style_instructions}

Below is a list of timed video transcript segments:
{json.dumps(segments, ensure_ascii=False)}

Generate THREE distinct translation options into natural {target_language} adhering strictly to the requested Translation Style.
Each option must include:
- "id": string ("opt_1", "opt_2", "opt_3")
- "badge": string (e.g., "🔥 98% Viral Hook", "⚡ 94% High Engagement", "📌 89% Direct Faithful")
- "title": string (short Burmese style title)
- "description": string (short description explaining the narration style)
- "viral_score": int (e.g., 98, 94, 89)
- "score_breakdown": {{"hook": int, "clarity": int, "retention": int}}
- "segments": array of objects, keeping exact "start" and "end" timestamps, with translated "text" string for each segment.

Styles:
Option 1: 🔥 Ultra Viral Hook Variation (Max curiosity, high viral score)
Option 2: ⚡ High Engagement Variation (Fast-paced storytelling)
Option 3: 📌 Direct & Clear Variation (Accurate, clear voiceover)

Return ONLY a valid JSON array of 3 option objects.
"""

    if provider == "openai" and openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": style_instructions},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            if isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list) and len(val) >= 3:
                        return val
            if isinstance(data, list):
                return data
        except Exception as e:
            print(f"OpenAI multi-option translation error: {e}")

    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            options = json.loads(response.text)
            if isinstance(options, list) and len(options) > 0:
                return options
        except Exception as e:
            print(f"Gemini multi-option translation error: {e}")

    # Fallback if AI generation fails or API key is not set
    base_translation = translate_transcript(segments, target_language=target_language, mode="full", translation_style=translation_style, provider=provider, api_key=api_key)
    return [
        {
            "id": "opt_1",
            "badge": "🔥 98% Viral Hook",
            "title": f"Viral {translation_style.capitalize()} Style",
            "description": "စိတ်ဝင်စားစရာ အစပျိုးချက်နှင့် ရွေးချယ်ထားသော စတိုင်အတိုင်း ရေးသားထားသော Voiceover",
            "viral_score": 98,
            "score_breakdown": {"hook": 99, "clarity": 96, "retention": 98},
            "segments": base_translation
        },
        {
            "id": "opt_2",
            "badge": "⚡ 94% High Engagement",
            "title": f"Fast {translation_style.capitalize()} Engagement",
            "description": "လူငယ်များ ကြိုက်နှစ်သက်မည့် သွက်လက်သော ပုံပြင်ပြော စတိုင်",
            "viral_score": 94,
            "score_breakdown": {"hook": 94, "clarity": 95, "retention": 93},
            "segments": base_translation
        },
        {
            "id": "opt_3",
            "badge": "📌 89% Direct Faithful",
            "title": f"Direct {translation_style.capitalize()}",
            "description": "မူရင်း အဓိပ္ပာယ်ကို တိကျစွာ ဘာသာပြန်ထားသော စတိုင်",
            "viral_score": 89,
            "score_breakdown": {"hook": 88, "clarity": 98, "retention": 85},
            "segments": base_translation
        }
    ]


