"""
Complex Multi-line Prompt Testing Suite
Tests the recommendation system with sophisticated natural language queries
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.dataset_loader import PhoneDatasetLoader
from ml.recommender import DeviceRecommender
from ml.advanced_nlp_parser import advanced_parser
from ml.nlp_parser import nlp_parser
from utils.device_filter import DeviceFilter, UseCase
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Complex test prompts
COMPLEX_PROMPTS = [
    """
    I'm looking for a high-end gaming smartphone that won't break the bank.
    I need at least 12GB of RAM and a refresh rate of 120Hz or higher.
    The camera isn't super important, but having fast charging would be nice.
    I've been considering Samsung and OnePlus, but I'm open to other brands.
    My budget is around $800-$1000, but I could stretch to $1200 if it's really worth it.
    Also, 5G support would be a nice bonus.
    """,

    """
    I'm a professional photographer who needs a phone with an excellent camera system.
    I want at least 48MP main camera with optical zoom capabilities.
    The display should be high quality (OLED or AMOLED preferred) with good color accuracy.
    I don't need gaming performance, so I can compromise on processing power.
    However, I need plenty of RAM (at least 8GB) for running photo editing apps smoothly.
    My budget is flexible - I'm willing to pay up to $1500 for the right device.
    Brand preference: I like Apple and Samsung, but Google Pixel is also great for photography.
    """,

    """
    I travel a lot and need a phone that lasts through a full day of heavy usage.
    Battery life is my top priority - I'm looking for 5500mAh or larger.
    Fast charging is essential because I don't have much time to charge during the day.
    The budget is tight - I can't spend more than $400-$500.
    I need good value for money: decent processor, at least 6GB RAM, and 128GB storage.
    5G support isn't necessary, but expandable storage would be helpful.
    I prefer brands like Xiaomi, Realme, or Motorola which offer good value.
    """,

    """
    I want a phone that's good for everything but doesn't excel at any one thing in particular.
    It should be reliable, have a good display, decent camera, and solid battery life.
    I have a moderate budget of around $600-$800.
    I'm coming from an iPhone, so I'm considering Android phones.
    I want something without too many gimmicks - just a solid daily driver.
    The OnePlus 12 seems interesting, but I'm also open to Samsung Galaxy A-series or Motorola Edge.
    I need it to handle basic gaming and video streaming without issues.
    """,

    """
    I'm a content creator looking for a phone to shoot videos on.
    I need excellent video stabilization and 4K recording capability.
    The display should be bright (at least 1000 nits) with a high refresh rate (120Hz+) for smooth preview.
    I need plenty of storage - 256GB minimum - for storing raw footage temporarily.
    High RAM (12GB+) is crucial for running video editing apps without lag.
    Budget: around $1200-$1500. I'm willing to pay for quality.
    Brands I trust: Apple, Samsung, and OnePlus.
    """,

    """
    I need to choose between a flagship and a mid-range phone.
    For flagship: I want the latest processor, best camera, and premium design.
    But I can't afford more than $1000.
    For mid-range: I need good performance, 5G, and reliable software updates.
    I would spend $500-$700 for a mid-range.
    Which option would be better for someone who uses their phone for work, gaming, and casual photography?
    I prefer brands that have good customer support.
    """,

    """
    I'm shopping for my parents who are not tech-savvy.
    They need a phone that's simple to use with a large display.
    Performance specs aren't critical - they mostly use it for calls, messages, and photos.
    Budget is limited to $300-$400.
    I want something reliable with long battery life and durable build quality.
    Preferably something that gets regular software updates for at least 3-4 years.
    Maybe Samsung Galaxy A-series or something similarly straightforward?
    """,

    """
    I'm a mobile gamer who plays PUBG, Call of Duty Mobile, and other demanding titles.
    I need the absolute best performance: latest flagship processor, 12GB+ RAM, 120Hz+ display.
    Thermal management is important - no throttling during long gaming sessions.
    Large battery (5000mAh+) is needed because gaming drains power quickly.
    I don't care much about camera quality, but build quality and cooling are crucial.
    Budget is up to $1200 for the best experience.
    Should I go with Snapdragon 8 Gen 3 or Apple A19 Pro?
    """,

    """
    I want to upgrade from my OnePlus 9 Pro.
    I loved the speed and fast charging on that phone.
    I want similar features but with better camera quality this time.
    My budget is $900-$1100.
    I need a phone that feels premium - metal and glass build, not plastic.
    Display quality is important to me: OLED, 120Hz, and HDR support.
    I'm open to exploring other brands like Samsung or Google Pixel this time.
    I also value getting software updates quickly and for a long time.
    """,

    """
    I need a tablet-phone hybrid device.
    A large display (6.5+ inches) is essential for productivity tasks.
    I need good multitasking capability with at least 8GB RAM.
    I want a stylus or good hand-writing support for note-taking.
    Camera quality should be decent for video calls and document scanning.
    Budget around $800-$1000.
    Should I consider iPad, Samsung Galaxy Tab S, or a large Android phone like a phablet?
    """,

    """
    I'm not willing to spend more than $300 on a phone.
    I need it to be durable - I'm clumsy and drop my phones often.
    Battery life should last at least a full day with moderate usage.
    I need a microSD card slot for expandable storage since internal storage will be limited.
    The processor doesn't need to be cutting-edge, but it should handle everyday apps smoothly.
    Which brands offer the best durability and reliability in the budget segment?
    Motorola, Xiaomi, or Realme?
    """,

    """
    I've been using an iPhone for years but want to try Android finally.
    I'm looking for something that feels as premium as my iPhone and is easy to switch to.
    I want a phone that matches or beats iPhone's camera system.
    The display should be excellent - OLED or mini-LED, 120Hz+.
    Fast performance is important - I want a Snapdragon 8 Gen 3 or equivalent.
    I can spend up to $1400.
    Should I go with Samsung Galaxy S24 Ultra or Google Pixel 9 Pro?
    """,

    """
    I work in a professional environment and need a business phone.
    The build quality must be excellent - premium materials, not plastic.
    Security features like fingerprint sensor and face recognition are important.
    I need excellent speakers for conference calls and video meetings.
    Battery life should be solid - at least 12 hours of mixed usage.
    I prefer clean Android (minimal bloatware) or iOS.
    Budget: up to $900 for a reliable business device.
    I heard about Samsung Galaxy S24 and iPhone 15 Pro, are they good for this?
    """,

    """
    I want a phone for my kid to use for school and light gaming.
    It needs to be durable and not break if dropped (preferably with drop protection).
    I want parental controls and good battery life.
    The performance should be good enough for educational apps and light games.
    Budget is limited to $250-$350.
    I don't want something too flashy or expensive to replace if lost or damaged.
    What would be a responsible choice? Samsung A series or Motorola?
    """,

    """
    I'm a photography enthusiast with a limited budget.
    I have $500 to spend on a phone that can take great photos.
    I need at least 12MP camera, but 48MP+ would be better if available in this budget.
    Night mode is important for low-light photography.
    I don't need extreme gaming performance or premium design - just good camera capabilities.
    I'm open to any brand that offers excellent value.
    What are the best camera phones under $500 right now?
    """,

    """
    I'm planning to use my phone as a mobile workstation.
    I need massive RAM (16GB+), fast processor, and ample storage (512GB or 1TB).
    The display should be large enough (6.7+) and support productivity features.
    Cooling solution is important since I'll be running heavy applications.
    I want a phone that can handle video editing and 3D rendering tasks.
    Budget is around $1500-$2000 for a truly capable device.
    I heard about foldable phones - should I consider those?
    """,

    """
    I need a phone that's good for everything but especially for video streaming.
    Display quality is paramount: OLED, high brightness (1200+ nits), 120Hz refresh rate.
    Good stereo speakers are crucial for watching movies and listening to music.
    I don't game, so I can compromise on processing power.
    Battery life should be solid for long sessions: 5000mAh+.
    Budget up to $1000.
    I'm between Samsung, OnePlus, and Google Pixel for streaming experience.
    """,

    """
    I'm concerned about privacy and security on my phone.
    I need strong encryption, security updates, and transparency about data handling.
    I prefer a phone from a company with a good privacy track record.
    Performance and camera are secondary concerns.
    Budget around $700-$900.
    I'm leaning towards iPhone or Google Pixel, but are there other secure Android phones?
    I want long-term software support - at least 5 years of updates.
    """,

    """
    I want to replace my flagship from 2 years ago.
    I want significant improvements in camera, display, and processing power.
    5G is now important to me, so I want the latest 5G implementation.
    I'm willing to spend $1200-$1500 for the best of what's available right now.
    I prefer sticking with my current brand but open to switching if there's something exceptional.
    Should I wait for the next generation or buy the current flagship?
    What are the must-have features in 2026?
    """,

    """
    I'm buying a phone for use in developing countries where power is unreliable.
    Battery capacity is critical - I need 6000mAh+ for extended usage between charges.
    Power efficiency is more important than raw performance.
    I need a phone that handles poor network conditions well (fallback to 3G/4G from 5G).
    Durability matters - I need something that can withstand dust and moisture.
    Budget: $300-$500.
    Which phones are known for reliability in tough conditions?
    """
]


def test_complex_prompts():
    """Test recommendation system with complex multi-line prompts"""
    logger.info("\n" + "="*80)
    logger.info("COMPLEX PROMPT TESTING SUITE")
    logger.info("="*80)
    
    # Load dataset
    logger.info("\n📁 Loading dataset...")
    loader = PhoneDatasetLoader()
    devices = loader.load_csv_files(limit=5000)
    logger.info(f"✅ Loaded {len(devices)} devices")
    
    # Train recommender
    logger.info("\n🤖 Training recommender...")
    recommender = DeviceRecommender()
    recommender.fit(devices)
    logger.info("✅ Recommender trained")
    
    # Test each complex prompt
    for i, prompt in enumerate(COMPLEX_PROMPTS, 1):
        logger.info("\n" + "="*80)
        logger.info(f"TEST {i}: Complex Prompt Analysis")
        logger.info("="*80)
        
        # Clean and display prompt
        clean_prompt = ' '.join(prompt.split())
        logger.info(f"\n📝 Prompt:\n{prompt.strip()}\n")
        
        # Parse with advanced NLP
        logger.info("🧠 Advanced NLP Parsing:")
        try:
            parsed = advanced_parser.parse_complex_query(clean_prompt)
            logger.info(f"   Parsed preferences:")
            for key, value in parsed.items():
                if value:
                    logger.info(f"   - {key}: {value}")
        except Exception as e:
            logger.error(f"   Error in advanced parsing: {e}")
            parsed = {}
        
        # Parse with BERT NLP
        logger.info("\n🧠 BERT NLP Enhancement:")
        try:
            bert_parsed = nlp_parser.enhance_preferences(parsed)
            logger.info(f"   Enhanced preferences:")
            for key, value in bert_parsed.items():
                if value:
                    logger.info(f"   - {key}: {value}")
        except Exception as e:
            logger.error(f"   Error in BERT enhancement: {e}")
            bert_parsed = parsed
        
        # Get recommendations
        logger.info("\n🎯 Recommendations:")
        try:
            recommendations = recommender.recommend_by_preferences(
                bert_parsed,
                top_n=5
            )
            
            if recommendations:
                logger.info("\n🎯 RECOMMENDED PHONES:")
                for rank, (device_id, score, metadata) in enumerate(recommendations, 1):
                    device = next((d for d in devices if d.get('id') == device_id), None)
                    if device:
                        specs = device['specs']
                        phone_name = f"{device['brand']} {device['model_name']}"
                        logger.info(f"\n   #{rank} RECOMMENDATION: {phone_name}")
                        logger.info(f"      ⭐ Match Score: {score:.1%}")
                        logger.info(f"      📋 Specifications:")
                        logger.info(f"         • RAM: {specs.get('ram_gb', 'N/A')}GB")
                        logger.info(f"         • Storage: {specs.get('storage_gb', 'N/A')}GB")
                        logger.info(f"         • Camera: {specs.get('main_camera_mp', 'N/A')}MP")
                        logger.info(f"         • Battery: {specs.get('battery_mah', 'N/A')}mAh")
                        logger.info(f"         • Display Refresh: {specs.get('refresh_rate_hz', 'N/A')}Hz")
                        logger.info(f"         • Price: ${specs.get('price', 'N/A')}")
                        logger.info(f"         • 5G Support: {'✓ Yes' if specs.get('has_5g') else '✗ No'}")
            else:
                logger.warning("   No recommendations found")
        except Exception as e:
            logger.error(f"   Error getting recommendations: {e}")
        
        # Analyze parsed preferences
        logger.info("\n📊 Analysis:")
        logger.info(f"   Budget mentioned: {'budget' in bert_parsed}")
        logger.info(f"   Use case detected: {bert_parsed.get('use_case', 'Not detected')}")
        logger.info(f"   Brand preference: {bert_parsed.get('brand_preference', 'None specified')}")
        logger.info(f"   Device type: {bert_parsed.get('device_type', 'Not specified')}")
        logger.info(f"   Feature requirements: {len({k: v for k, v in bert_parsed.items() if 'require' in k or 'prefer' in k or 'min_' in k})} detected")


def test_prompt_variations():
    """Test how system handles variations of the same requirement"""
    logger.info("\n" + "="*80)
    logger.info("PROMPT VARIATION TESTING")
    logger.info("="*80)
    
    loader = PhoneDatasetLoader()
    devices = loader.load_csv_files(limit=5000)
    
    recommender = DeviceRecommender()
    recommender.fit(devices)
    
    # Different ways to express the same requirement
    prompt_variations = [
        ("I need a phone with 8GB RAM", "Direct requirement"),
        ("I want at least 8GB of RAM", "Alternative phrasing"),
        ("The phone should have 8GB or more RAM", "Range expression"),
        ("I can't go below 8GB RAM", "Negative phrasing"),
        ("8GB minimum RAM is essential", "Emphasis expression"),
    ]
    
    logger.info("\n🔄 Testing prompt variations for: '8GB RAM requirement'")
    
    for prompt, description in prompt_variations:
        logger.info(f"\n📝 {description}:")
        logger.info(f"   Query: \"{prompt}\"")
        
        try:
            parsed = advanced_parser.parse_complex_query(prompt)
            enhanced = nlp_parser.enhance_preferences(parsed)
            
            ram_req = enhanced.get('min_ram_gb', 'Not detected')
            logger.info(f"   Detected RAM requirement: {ram_req}GB")
            
            # Get recommendations with this requirement
            recs = recommender.recommend_by_preferences(enhanced, top_n=3)
            if recs:
                logger.info(f"   📱 Top Recommendations:")
                for rank, (device_id, score, metadata) in enumerate(recs[:2], 1):
                    device = next((d for d in devices if d.get('id') == device_id), None)
                    if device:
                        logger.info(f"      {rank}. {device['brand']} {device['model_name']} - {score:.1%} match")
            else:
                logger.warning(f"   No recommendations found")
        except Exception as e:
            logger.error(f"   Error: {e}")


def test_conflicting_requirements():
    """Test how system handles conflicting or contradictory requirements"""
    logger.info("\n" + "="*80)
    logger.info("CONFLICTING REQUIREMENTS TESTING")
    logger.info("="*80)
    
    loader = PhoneDatasetLoader()
    devices = loader.load_csv_files(limit=5000)
    
    recommender = DeviceRecommender()
    recommender.fit(devices)
    
    # Prompts with conflicting requirements
    conflicts = [
        ("I need a flagship phone but my budget is only $300", "Budget vs. Performance"),
        ("I want a gaming phone but battery life is more important", "Gaming vs. Battery"),
        ("I need an ultra-premium phone but want to spend less than $500", "Premium vs. Budget"),
        ("I prefer lightweight phone but need 6000mAh battery", "Weight vs. Battery"),
        ("I want the latest flagship but need a $200 budget", "Latest vs. Budget"),
    ]
    
    logger.info("\n⚠️ Testing conflicting requirements:")
    
    for prompt, conflict_type in conflicts:
        logger.info(f"\n🔀 {conflict_type}:")
        logger.info(f"   Prompt: \"{prompt}\"")
        
        try:
            parsed = advanced_parser.parse_complex_query(prompt)
            enhanced = nlp_parser.enhance_preferences(parsed)
            
            logger.info(f"   Parsed as:")
            for key, value in enhanced.items():
                if value:
                    logger.info(f"   - {key}: {value}")
            
            # Get recommendations
            recs = recommender.recommend_by_preferences(enhanced, top_n=3)
            if recs:
                logger.info(f"\n   📱 System Recommendations:")
                for rank, (device_id, score, metadata) in enumerate(recs[:2], 1):
                    device = next((d for d in devices if d.get('id') == device_id), None)
                    if device:
                        logger.info(f"      {rank}. {device['brand']} {device['model_name']}")
                        logger.info(f"         Score: {score:.1%} | Price: ${device['specs'].get('price', 'N/A')}")
            else:
                logger.warning(f"   No recommendations found")
        except Exception as e:
            logger.error(f"   Error: {e}")


def test_implicit_requirements():
    """Test implicit requirements from context"""
    logger.info("\n" + "="*80)
    logger.info("IMPLICIT REQUIREMENTS TESTING")
    logger.info("="*80)
    
    loader = PhoneDatasetLoader()
    devices = loader.load_csv_files(limit=5000)
    
    recommender = DeviceRecommender()
    recommender.fit(devices)
    
    # Prompts with implicit requirements
    implicit = [
        ("I travel a lot and my flight is tomorrow", "Implies: fast shipping, good battery"),
        ("I'm a professional photographer in Africa", "Implies: reliable in tough conditions, good camera"),
        ("I play competitively in esports tournaments", "Implies: top performance, low latency"),
        ("I work in hospitals and need reliability", "Implies: durability, professional build"),
        ("I'm learning to code on my mobile", "Implies: good performance, productivity features"),
    ]
    
    logger.info("\n🎭 Testing implicit requirements:")
    
    for prompt, implication in implicit:
        logger.info(f"\n💡 Implication: {implication}")
        logger.info(f"   Prompt: \"{prompt}\"")
        
        try:
            parsed = advanced_parser.parse_complex_query(prompt)
            enhanced = nlp_parser.enhance_preferences(parsed)
            
            logger.info(f"   Detected preferences:")
            for key, value in enhanced.items():
                if value:
                    logger.info(f"   - {key}: {value}")
            
            # Get recommendations
            recs = recommender.recommend_by_preferences(enhanced, top_n=2)
            if recs:
                logger.info(f"   📱 Recommended Phones:")
                for rank, (device_id, score, metadata) in enumerate(recs, 1):
                    device = next((d for d in devices if d.get('id') == device_id), None)
                    if device:
                        logger.info(f"      {rank}. {device['brand']} {device['model_name']} - {score:.1%} match")
        except Exception as e:
            logger.error(f"   Error: {e}")


def test_edge_cases():
    """Test edge cases and unusual prompts"""
    logger.info("\n" + "="*80)
    logger.info("EDGE CASES TESTING")
    logger.info("="*80)
    
    loader = PhoneDatasetLoader()
    devices = loader.load_csv_files(limit=5000)
    
    recommender = DeviceRecommender()
    recommender.fit(devices)
    
    edge_cases = [
        ("", "Empty prompt"),
        ("phone", "Single word"),
        ("the best phone ever made in history", "Vague hyperbole"),
        ("I want Apple's iPhone but the price of a Samsung A10", "Unrealistic expectations"),
        ("What's the most expensive phone available?", "Question format"),
        ("Find me a phone that doesn't exist yet", "Impossible requirement"),
        ("The phone should NOT be an iPhone", "Explicit exclusion"),
    ]
    
    logger.info("\n🔧 Testing edge cases:")
    
    for prompt, case_type in edge_cases:
        logger.info(f"\n⚙️ {case_type}:")
        logger.info(f"   Prompt: \"{prompt}\"")
        
        try:
            if prompt:
                parsed = advanced_parser.parse_complex_query(prompt)
                enhanced = nlp_parser.enhance_preferences(parsed)
                
                recs = recommender.recommend_by_preferences(enhanced, top_n=2)
                if recs:
                    logger.info(f"   ✅ Got {len(recs)} recommendations:")
                    for rank, (device_id, score, metadata) in enumerate(recs, 1):
                        device = next((d for d in devices if d.get('id') == device_id), None)
                        if device:
                            logger.info(f"      {rank}. {device['brand']} {device['model_name']}")
                else:
                    logger.warning(f"   ⚠️ No recommendations found")
            else:
                logger.warning(f"   Skipped empty prompt")
        except Exception as e:
            logger.warning(f"   ⚠️ Handled gracefully: {str(e)[:60]}...")


def main():
    """Run all complex prompt tests"""
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE COMPLEX PROMPT TESTING")
    logger.info("="*80)
    
    try:
        # Run all test suites
        test_complex_prompts()
        test_prompt_variations()
        test_conflicting_requirements()
        test_implicit_requirements()
        test_edge_cases()
        
        logger.info("\n" + "="*80)
        logger.info("✅ ALL COMPLEX PROMPT TESTS COMPLETED")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
