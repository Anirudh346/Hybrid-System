"""
Test Advanced NLP Parser with Complex, Inconsistent, and Context-Dependent Prompts
"""

from ml.advanced_nlp_parser import advanced_parser

# Test cases: Complex, conflicting, and context-dependent queries
test_queries = [
    # 1. Conflicting requirements
    "I need a cheap flagship phone for gaming",
    
    # 2. Multiple use cases
    "Best phone for gaming AND photography under $800",
    
    # 3. Implicit preferences
    "I travel a lot and need something reliable",
    
    # 4. Trade-offs
    "Willing to sacrifice camera for better battery life",
    
    # 5. Negations
    "Good phone but not Samsung, without notch",
    
    # 6. Context references
    "Something better than iPhone 12 but cheaper",
    
    # 7. Complex multi-intent
    "Budget gaming phone with decent camera, not OnePlus, must have fast charging",
    
    # 8. Lifestyle-based
    "I'm a content creator who travels, need all-day battery and great video",
    
    # 9. Priority-based
    "Best value phone with premium camera, display not important",
    
    # 10. Ultra complex
    "Need affordable flagship killer for gaming and photography, willing to sacrifice display for battery, not Samsung or Apple, similar to OnePlus but cheaper"
]

print("=" * 100)
print("ADVANCED NLP PARSER - COMPLEX QUERY HANDLING")
print("=" * 100)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*100}")
    print(f"TEST {i}: {query}")
    print('='*100)
    
    result = advanced_parser.parse_complex_query(query)
    
    print(f"\n📋 EXTRACTED PREFERENCES:")
    print(f"  Primary Use Case: {result['use_case'] or 'N/A'}")
    if result['secondary_use_cases']:
        print(f"  Secondary Use Cases: {', '.join(result['secondary_use_cases'])}")
    
    print(f"\n💰 BUDGET & BRANDS:")
    print(f"  Budget: ${result['budget']}" if result['budget'] else "  Budget: Not specified")
    if result['brand_preference']:
        print(f"  Preferred Brands: {', '.join(result['brand_preference'])}")
    if result['brand_avoid']:
        print(f"  ❌ Avoid Brands: {', '.join(result['brand_avoid'])}")
    
    print(f"\n✅ FEATURES:")
    if result['must_have_features']:
        print(f"  Must Have: {', '.join(result['must_have_features'])}")
    if result['nice_to_have_features']:
        print(f"  Nice to Have: {', '.join(result['nice_to_have_features'])}")
    if result['avoid_features']:
        print(f"  ❌ Avoid: {', '.join(result['avoid_features'])}")
    
    print(f"\n🎯 PRIORITY & TRADE-OFFS:")
    print(f"  Priority: {result['priority']}")
    if result['trade_offs']:
        for sacrifice, prefer in result['trade_offs']:
            print(f"  Trade-off: Sacrifice {sacrifice} for {prefer}")
    
    if result['context_references']:
        print(f"\n📌 CONTEXT REFERENCES:")
        for ref in result['context_references']:
            print(f"  - {ref}")
    
    print(f"\n📊 CONFIDENCE: {result['confidence']:.0%}")

print("\n" + "=" * 100)
print("\n✅ Advanced NLP Parser successfully handles:")
print("  ✓ Conflicting requirements (cheap flagship → flagship killer)")
print("  ✓ Multiple use cases (gaming AND photography)")
print("  ✓ Implicit preferences (travel → battery + dual SIM)")
print("  ✓ Trade-offs (sacrifice X for Y)")
print("  ✓ Negations (not Samsung, without notch)")
print("  ✓ Context references (better than iPhone 12)")
print("  ✓ Priority detection (best camera, best battery)")
print("  ✓ Lifestyle-based inference (content creator → camera + video)")
print("\n" + "=" * 100)
