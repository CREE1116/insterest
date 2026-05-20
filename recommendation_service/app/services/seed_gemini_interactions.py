import asyncio
import json
import uuid
import random
import httpx
from sqlalchemy import text
from app.db.session import AsyncSessionLocal
from app.core.config import settings
from app.core.logging import logger

async def seed_gemini_interactions_service(db) -> dict:
    logger.info("🤖 Starting Dynamic Gemini-based Multi-Persona Generation and Interaction seeding...")
    
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        logger.error("❌ GEMINI_API_KEY is not configured in settings.")
        return {"status": "error", "message": "GEMINI_API_KEY is not configured in settings."}

    # 1. Fetch posts from DB
    res = await db.execute(text("SELECT post_id, content_text FROM search.post_vectors ORDER BY created_at DESC LIMIT 150"))
    rows = res.all()
    if not rows:
        logger.warning("⚠️ No posts found in search.post_vectors.")
        return {"status": "error", "message": "No posts found in database. Please run dummy seeding first."}

    posts = []
    for r in rows:
        post_id = str(r[0])
        meta = r[1]
        caption = meta.get("caption", "") if isinstance(meta, dict) else ""
        tags = ", ".join(meta.get("tags", [])) if isinstance(meta, dict) and isinstance(meta.get("tags"), list) else ""
        posts.append({"id": post_id, "caption": caption, "tags": tags})

    # Prepare post text list for prompt
    post_descriptions = []
    for p in posts:
        post_descriptions.append(f"- ID: {p['id']} | Caption: {p['caption']} | Tags: {p['tags']}")
    posts_text = "\n".join(post_descriptions)

    # 2. Query Gemini to generate 50 personas and their likes
    client = httpx.AsyncClient(timeout=60.0)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    prompt = f"""
You are a creative synthetic data generator for a recommendation system.
We have 150 posts in our feed database:
{posts_text}

Task:
1. Generate 50 highly diverse user personas with unique names (usernames) and descriptions representing various modern lifestyles (e.g. foodies, pet lovers, fitness runners, cafe hoppers, travelers, minimalists, software developers, movie buffs, fashion designers, art collectors, etc.).
2. For each persona, review the posts above and select between 8 and 15 post IDs that match their interests to simulate "likes".
3. Return the result in a JSON object matching this schema:
{{
  "personas": [
    {{
      "name": "Username",
      "desc": "Short interest description",
      "likes": ["uuid_string_1", "uuid_string_2"]
    }}
  ]
}}

Return ONLY the raw JSON object. Do not wrap in markdown or prefix with comments.
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    personas_data = []
    try:
        logger.info("🧠 Requesting 50 dynamic personas and likes from Gemini API...")
        response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            resp_json = response.json()
            text_out = resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            data = json.loads(text_out)
            personas_data = data.get("personas", [])
        else:
            logger.error(f"❌ Gemini API returned status {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"❌ Error calling Gemini API: {str(e)}")

    # Fallback if Gemini failed
    if not personas_data:
        logger.warning("⚠️ Gemini dynamic generation failed. Falling back to synthetic programmatic personas.")
        # Programmatic generation of 50 personas
        themes = [
            ("CatLover", "Obsessed with cats and fluffy kittens"),
            ("TechGuru", "Software development, coding setups, and mechanical keyboards"),
            ("CafeHopper", "Loves vintage coffee shops, latte art, and sweet desserts"),
            ("SeoulFoodie", "Korean food, spicy tteokbokki, and street markets"),
            ("FitnessBro", "Gym workouts, running marathons, and athletic clothing"),
            ("NatureLover", "Camping in mountains, hiking, and forest streams"),
            ("HomeDeco", "Minimalist bedroom designs and cozy living spaces"),
            ("FashionIcon", "Streetwear fashion, luxury bags, and modern outfit ideas"),
            ("ArtCurator", "Gallery exhibits, modern paintings, and aesthetic sketches"),
            ("MovieBuff", "Cinematic trailers, theater experiences, and pop culture")
        ]
        for i in range(50):
            theme = random.choice(themes)
            name = f"{theme[0]}_{i+1}"
            desc = theme[1]
            # Select random matching posts
            liked = [uuid.UUID(p["id"]) for p in random.sample(posts, min(random.randint(8, 15), len(posts)))]
            personas_data.append({
                "name": name,
                "desc": desc,
                "likes": [str(l) for l in liked]
            })

    # 3. Save personas and likes in DB
    seeded_count = 0
    persona_results = []
    
    for p_info in personas_data:
        name = p_info.get("name", "User")
        desc = p_info.get("desc", "Interest")
        likes = p_info.get("likes", [])
        
        # Consistent UUID generation based on name
        user_id = uuid.uuid5(uuid.NAMESPACE_DNS, name)
        
        # Filter valid likes
        valid_liked = []
        for lid in likes:
            try:
                valid_liked.append(uuid.UUID(lid))
            except ValueError:
                continue
                
        inserted = 0
        for i, post_id in enumerate(valid_liked):
            try:
                await db.execute(text("""
                    INSERT INTO interaction.likes (user_id, post_id, created_at)
                    VALUES (:u, :p, NOW() - INTERVAL '1 second' * :offset)
                    ON CONFLICT DO NOTHING
                """), {"u": user_id, "p": post_id, "offset": len(valid_liked) - i})
                inserted += 1
            except Exception as e:
                logger.error(f"   ❌ Failed to insert like: {e}")
                
        seeded_count += inserted
        persona_results.append({
            "name": name,
            "desc": desc,
            "user_id": str(user_id),
            "liked_count": inserted
        })

    await db.commit()
    await client.aclose()
    
    logger.info(f"🎉 Dynamic seeding complete. Created {seeded_count} virtual likes across {len(personas_data)} dynamic Gemini personas.")
    return {
        "status": "success",
        "personas_count": len(personas_data),
        "total_likes_seeded": seeded_count,
        "details": persona_results
    }
