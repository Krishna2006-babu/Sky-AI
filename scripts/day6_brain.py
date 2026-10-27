# --- THE KNOWLEDGE BASE ---
# We store thresholds and recommendations in one place for easy updating.
SKIN_GUIDE = {
    "redness": {
        "limit": 20, 
        "status": "🔴 Irritated/Inflamed",
        "ingredients": ["Niacinamide", "Aloe Vera", "Centella"],
        "action": "Avoid harsh scrubs and use a calming moisturizer."
    },
    "oil": {
        "limit": 8.0,
        "status": "💧 High Sebum (Oily)",
        "ingredients": ["Salicylic Acid (BHA)", "Clay Mask"],
        "action": "Use a foaming cleanser and oil-free hydration."
    },
    "texture": {
        "limit": 150.0,
        "status": "🧩 High Texture/Pores",
        "ingredients": ["AHA (Glycolic Acid)", "Retinol"],
        "action": "Incorporate gentle exfoliation 2x a week."
    }
}

def generate_report(r_score, o_percent, t_score):
    final_issues = []
    final_ingredients = set() # 'set' ensures no duplicates

    print("\n" + "="*40)
    print("📋 SKY-AI DIAGNOSTIC REPORT")
    print("="*40)

    # Logic for Redness
    if r_score > SKIN_GUIDE["redness"]["limit"]:
        final_issues.append(SKIN_GUIDE["redness"]["status"])
        final_ingredients.update(SKIN_GUIDE["redness"]["ingredients"])
        print(f"Advice: {SKIN_GUIDE['redness']['action']}")

    # Logic for Oil
    if o_percent > SKIN_GUIDE["oil"]["limit"]:
        final_issues.append(SKIN_GUIDE["oil"]["status"])
        final_ingredients.update(SKIN_GUIDE["oil"]["ingredients"])
        print(f"Advice: {SKIN_GUIDE['oil']['action']}")

    # Logic for Texture
    if t_score > SKIN_GUIDE["texture"]["limit"]:
        final_issues.append(SKIN_GUIDE["texture"]["status"])
        final_ingredients.update(SKIN_GUIDE["texture"]["ingredients"])
        print(f"Advice: {SKIN_GUIDE['texture']['action']}")

    return final_issues, list(final_ingredients)

# TEST DATA (Use your actual scores from Day 5 here!)
my_redness = 22   # From your color analysis
my_oil = 5.2      # From your HSV analysis
my_texture = 165  # From your Laplacian analysis

issues, products = generate_report(my_redness, my_oil, my_texture)

print(f"\n🔎 DETECTED ISSUES: {', '.join(issues) if issues else 'None (Healthy!)'}")
print(f"🧪 KEY INGREDIENTS: {', '.join(products)}")
print("="*40)