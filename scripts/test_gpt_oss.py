import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="",
)

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b:cerebras",
    messages=[
        {
            "role": "user",
            "content": """You classify Google reviews into one of four exclusive categories:
                        Trustworthy: Genuine experience at the business.
                        Advertisement: Promo/marketing, owner self-post, coupon links.
                        Irrelevant Content: Off-topic, spam, wrong place, meaningless.
                        Rant without visit: Complaints/opinions with no evidence of visiting/using the business.
                        If in doubt, choose the stricter non-Trustworthy label.
                        Instructions
                        Input: Two JSONs — business (metadata) and review (user review), linked by gmap_id.
                        Cross-check that the review plausibly matches the business (category, services, etc.).
                        Star rating/typos don’t affect trustworthiness.


                        Input: business metadata JSON
                        {'name': 'Purple Peanut',
                        'address': 'Purple Peanut, 2357 Whitesburg Dr # D, Huntsville, AL 35801',
                        'gmap_id': '0x8862134e67ff5c87:0x38b5e2ae99cd1fcf',
                        'description': None,
                        'latitude': 34.7131627,
                        'longitude': -86.57404149999999,
                        'category': ['Boutique'],
                        'avg_rating': 4.6,
                        'num_of_reviews': 17,
                        'price': None,
                        'hours': [['Thursday', 'Closed'],
                        ['Friday', '10AM–5PM'],
                        ['Saturday', '10AM–3PM'],
                        ['Sunday', 'Closed'],
                        ['Monday', 'Closed'],
                        ['Tuesday', '10AM–5PM'],
                        ['Wednesday', '10AM–5PM']],
                        'MISC': {'Service options': ['In-store shopping', 'Delivery'],
                        'Accessibility': ['Wheelchair accessible entrance'],
                        'Planning': ['Quick visit']},
                        'state': 'Closed ⋅ Opens 10AM Fri',
                        'relative_results': ['0x886269389730ce8b:0xedffba4037968914',
                        '0x88626cba32c55791:0xb2b825009c312370',
                        '0x88626b52b85fa957:0x418f198557020e80',
                        '0x88626622a77eec85:0x71b49ce40b0a37f5',
                        '0x886269389730ce8b:0x1a711b728215c00a'],
                        'url': 'https://www.google.com/maps/place//data=!4m2!3m1!1s0x8862134e67ff5c87:0x38b5e2ae99cd1fcf?authuser=-1&hl=en&gl=us'
                        }

                        Review JSON
                        {
                        'user_id': '114043824230907811356',
                        'time': 1597168272670,
                        'rating': 5,
                        'text': 'Very Personable staff! Beautiful and clean environment. I will definitely become a regular customer!!',
                        'pics': None,
                        'resp': None,
                        'gmap_id': '0x8862134e67ff5c87:0x38b5e2ae99cd1fcf'
                        }

                        Pick one label only to classify the review.
                        Output in the below JSON format only:

                        {
                        "role": "assistant",
                        "content: {
                            "gmap_id": "<string>",
                            "label": "Trustworthy | Advertisement | Irrelevant Content | Rant without visit",
                            "confidence": 0.0,
                            "rationale": "<short reason>"
                            }
                        }
                    """,
            "reasoning": "low"
        }
    ],
)

print(completion.choices[0].message)
