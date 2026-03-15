---
name: security-camera-analysis
description: Analyze security camera images for threat assessment. Use when receiving surveillance footage, motion detection images, or security alerts. Switches to Gemini for vision analysis, then DeepSeek for Notion logging and Telegram alerts.
license: MIT
metadata:
  author: zawar
  version: "2.1"
---

# Security Camera Threat Analysis

When triggered with a security camera image, follow this EXACT pipeline in order:

## Step 1: View and Analyze the Image

**Note:** This skill requires a vision-capable model. When called from server.py, the model is specified via CLI flag `-m nous:gemini-3-flash`.

Use your vision tool to look at the image file provided. Assess:

1. **Weapons**: Guns, knives, bats, crowbars, threatening objects
2. **Face concealment**: Masks, balaclavas, hoodies hiding face, sunglasses at night
3. **Suspicious behavior**: Lurking, peeking in windows, trying doors, crouching, running
4. **Clothing**: All dark at night, tactical gear, gloves in warm weather
5. **Body language**: Aggressive posture, hiding something, casing property
6. **Context**: Time of day, number of people, approach pattern

You MUST actually view the image — never guess based on metadata alone.

## Step 2: Score the Threat

| Level | Score | When |
|-------|-------|------|
| NONE | 0 | Known person, delivery driver, obvious non-threat |
| LOW | 1-3 | Unknown person, normal behavior (walking past, neighbor) |
| MEDIUM | 4-6 | Unknown with mildly suspicious indicators (lingering, looking around) |
| HIGH | 7-8 | Face concealed, suspicious approach, trying to hide from camera |
| CRITICAL | 9-10 | Visible weapon, break-in attempt, multiple masked individuals |

**Scoring rules:**
- Concealed face = minimum HIGH (7)
- Person at night near door without package = minimum MEDIUM (4)
- Multiple unknown people at night = minimum MEDIUM (5)
- Any weapon visible = CRITICAL (10)
- Delivery driver with package/uniform = LOW (1)

Record these values from your analysis:
- threat_level (string)
- score (number 0-10)
- summary (one sentence)
- description (detailed: gender, age, build, clothing, what they are doing)
- weapons_detected (true/false)
- face_concealed (true/false)
- suspicious_behavior (true/false)
- people_count (number)
- recommendation (what homeowner should do)

## Step 3: Push to Notion

Use the `notion` skill or `execute_code` to create an entry in the Notion database (ID from env var `NOTION_DATABASE_ID`).

**Working Property Mapping:**
- **Alert** (title): `{threat_level} — {camera} — {timestamp}`
- **Threat Level** (select): `threat_level`
- **Score** (number): `score`
- **Description** (rich_text): `description`
- **Camera** (rich_text): `camera`
- **Image** (url): `http://[SERVER_IP]:8899/images/FILENAME` (Get SERVER_IP from current environment)
- **Time** (date): `timestamp` (ISO format)

## Step 4: Send Telegram Notification

If threat level is **MEDIUM or above** (score >= 4), you MUST send a formatted alert to the user's Telegram (chat ID from env var `TELEGRAM_CHAT_ID`, formatted as `telegram:{TELEGRAM_CHAT_ID}`). Use double-spacing between paragraphs to ensure readability in the Telegram UI.

**IMPORTANT:** Do NOT include the detailed person description in the Telegram notification. Keep the full description for Notion only.

**Detailed Message Format:**
```markdown
[EMOJI] SECURITY ALERT: [THREAT_LEVEL] (Score: [SCORE]/10)


📷 Camera: [CAMERA_NAME]

🕐 Time: [TIMESTAMP]


📋 Summary: [ONE_SENTENCE_SUMMARY]


🔍 Observations:
• Weapons: [YES/NO]
• Face concealed: [YES/NO]
• Suspicious behavior: [YES/NO]
• People count: [COUNT]


💡 Recommendation: [ACTIONABLE_ADVICE]


📝 Logged to Notion ✓

MEDIA:[LOCAL_IMAGE_PATH]
```

**Implementation via Tool:**
Use the `schedule_cronjob` tool with `deliver="telegram:{TELEGRAM_CHAT_ID}"` (read `TELEGRAM_CHAT_ID` from environment) and an immediate one-shot schedule of "1m" (1 minute from now) for near-instant delivery. This is the minimum scheduling delay supported by the cronjob system.

**CRITICAL:** The `MEDIA:` tag must be on its own line at the very end of the prompt for the image to attach correctly to the Telegram bubble. Use `\n\n` between each section for the required paragraph spacing.

**Reminder:** Telegram alert fields are limited to Camera, Time, Summary, Observations, Recommendation, and the image attachment.

## Rules

- ALWAYS view the actual image in Step 1 — never skip vision
- Keep all analysis data in memory throughout the workflow
- ALWAYS push to Notion for every detection (even NONE/LOW)
- ONLY send Telegram alert for MEDIUM, HIGH, CRITICAL
- If vision fails, say so and default to MEDIUM threat with recommendation to check manually
- Be specific in person descriptions — gender, age estimate, build, clothing, hair, distinguishing features
- Keep recommendations actionable: "Turn on lights", "Check live feed", "Call authorities"
- Complete the full pipeline as fast as possible — user is waiting
