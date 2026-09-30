"""
Election Dataset Generator
Generates a realistic social media dataset for election sentiment analysis.
Contains authentic-style tweets, varied lengths, hashtags, colloquial expressions,
and realistic voter discourse for robust text mining.
"""

import os
import random
import pandas as pd
from datetime import datetime, timedelta

def generate_election_dataset(output_path, num_records=1500):
    random.seed(42)
    
    candidates = [
        {"name": "Candidate A", "party": "Alliance for Progress (AFP)", "handle": "@CandidateA"},
        {"name": "Candidate B", "party": "National Democratic Union (NDU)", "handle": "@CandidateB"},
        {"name": "Candidate C", "party": "Reform & Integrity Party (RIP)", "handle": "@CandidateC"}
    ]
    
    topics = [
        "Economy & Jobs",
        "Healthcare",
        "Education",
        "Infrastructure & Green Energy",
        "Governance & Ethics",
        "National Security"
    ]
    
    positive_phrases = [
        "{candidate} delivered an outstanding speech on {topic} today! Truly inspiring vision for our future. #Election2026 #Leadership",
        "Impressive economic plan presented by {candidate}! Lower inflation and 500k new green jobs is exactly what we need. {handle} #Growth",
        "Just watched the town hall. {candidate} demonstrated remarkable grasp of {topic}. They have my full support! #VoterChoice",
        "Huge turnout at the {candidate} rally today! The energy is electric and the momentum is unstoppable! #Election2026",
        "Finally a leader who prioritizes {topic}. {candidate}'s proposed reforms will uplift working families across the country! {handle}",
        "The debate performance of {candidate} was calm, articulate, and completely decisive on {topic}. Outstanding! #DebateWinner",
        "Proud to endorse {candidate}! A visionary roadmap for technology innovation, education, and social welfare. #ForwardTogether",
        "Healthcare accessibility for all citizens! {candidate} has consistently fought for medical reforms. Thank you {handle}!",
        "Historic infrastructure bill proposed by {candidate}. Investing in high-speed rail and renewable grid is visionary! #GreenFuture",
        "I was undecided until I read {candidate}'s policy paper on {topic}. Transparent, pragmatic, and data-driven! #SmartGovernance",
        "Strong leadership shown by {candidate} during the national crisis. Integrity and compassion in action! #TrueLeader",
        "Youth employment initiatives by {candidate} are groundbreaking. Giving the next generation real opportunities! {handle} #YouthVote",
        "Listening to the concerns of ordinary citizens! {candidate} spent hours at the community center answering tough questions on {topic}.",
        "Education funding increase without raising middle-class taxes? {candidate} actually found a viable budget surplus solution!",
        "Ethical governance and complete anti-corruption measures! Exactly why {candidate} deserves to lead this nation! #CleanPolitics",
        "Loving {candidate}'s honest take on {topic}. Real solutions rather than empty slogans.",
        "Yes! Great to see {candidate} gaining strong endorsements from grassroots organizations today.",
        "Solid arguments from {handle} tonight. The plan for {topic} is well thought out and realistic.",
        "Inspiring words from {candidate} at the community rally. Let's make this win happen!",
        "Very optimistic after hearing {candidate}'s interview. Sensible policies on {topic} for all families.",
        "Kudos to {candidate} for addressing rural healthcare and small businesses head on!",
        "Best candidate in the race by far. {handle} understands our economic challenges.",
        "Clear vision, dignity, and practical execution. Rooting for {candidate} in the upcoming vote!",
        "A refreshing speech on democratic renewal and ethics from {candidate}. Count me in!"
    ]
    
    negative_phrases = [
        "Deeply disappointed by {candidate}'s vague answers on {topic}. Zero concrete plans to tackle the crisis! {handle} #DebateFail",
        "{candidate}'s latest proposal will drive inflation even higher. Irresponsible spending without any fiscal discipline! #EconomicWarning",
        "How can anyone trust {candidate} on {topic} when their voting record shows the exact opposite for the past decade? #Hypocrisy",
        "Major blunders in today's press conference by {candidate}. Completely out of touch with the struggles of working people! #Election2026",
        "The proposed cuts to {topic} by {candidate} will devastate rural communities. Completely unacceptable! {handle} #VoterDiscontent",
        "Another empty campaign promise from {candidate}. Talk is cheap, but where is the actual funding for {topic}? #BrokenPromises",
        "Watching {candidate} dodge straightforward questions on {topic} was painful. Lack of transparency is alarming! #AccountabilityNow",
        "Terrible debate showing for {candidate}. Aggressive, defensive, and lacking any coherent policy vision on {topic}. #DebateLoss",
        "{candidate}'s track record on corruption and ethical conduct is deeply concerning. Voters deserve better! #EthicsMatter",
        "Rising unemployment numbers prove {candidate}'s economic strategy has failed miserably. Time for a change of direction! #JobsCrisis",
        "Ignoring public opinion on {topic}! {candidate} continues to serve special interests rather than ordinary families. {handle}",
        "Unrealistic claims from {candidate}! You cannot simultaneously cut revenues and promise massive subsidies in {topic}. Pure fantasy!",
        "Severe mismanagement in previous terms makes {candidate} unfit to oversee national {topic}. #NoConfidence",
        "Weak stance on national defense and border security from {candidate}. Compromising our national safety is non-negotiable!",
        "The tax burden on small businesses under {candidate}'s scheme would shut down thousands of local enterprises! #SaveSmallBiz",
        "I cannot believe people are falling for {candidate}'s false narratives on {topic}. Do your research!",
        "Total trainwreck of an interview from {handle}. Avoided every tough question about campaign financing.",
        "Cost of living is through the roof and {candidate} offers zero practical solutions. What a joke.",
        "Broken promises again. Remember what {candidate} said about {topic} two years ago? Never happened.",
        "Shocking lack of empathy from {candidate} today. Clearly disconnected from ordinary realities.",
        "Wasteful spending, no fiscal restraint, and reckless slogans from {handle}. Hard pass!",
        "The latest policy paper from {candidate} reads like a wish list with no real budget arithmetic.",
        "Very poor response from {candidate} regarding the corruption allegations. Silence speaks volumes.",
        "Disagree completely with {handle}'s authoritarian approach to public education and local governance."
    ]
    
    neutral_phrases = [
        "Election Update: {candidate} will hold a policy address on {topic} this Thursday at 3 PM EST. Coverage streaming live on channel 4. #ElectionUpdate",
        "New voter survey released today shows {candidate} holding a 48% approval rating among independent voters. #PollTracker #Data",
        "Both parties met today to finalize the debate rules covering {topic}, foreign policy, and federal budget allocations. #Campaign2026",
        "{candidate} submitted the official campaign expenditure statement to the election commission yesterday afternoon. {handle}",
        "Voter registration deadline is in two weeks. Check your polling precinct and local ballot measures online. #VoterInfo #Democracy",
        "Analysis of the three candidate manifestos regarding {topic}: A comparison of tax rates, timeline, and projected GDP impact. #PolicyReview",
        "{candidate} met with university faculty representatives to discuss federal grants for scientific research and {topic}. #EducationNews",
        "Live coverage: Press conference by {candidate} answering questions regarding recent bipartisan committee findings on {topic}.",
        "Polling stations will open at 7:00 AM on election day. Early mail-in voting starts next Monday across all counties. #ElectionGuide",
        "Statistical breakdown: Comparing historical voter turnout trends with early voting statistics in key swing districts. #ElectionMetrics",
        "The parliamentary committee report on {topic} was reviewed by {candidate}'s advisory team earlier today. #PolicyReport",
        "Campaign schedule: {candidate} is scheduled to visit three regional manufacturing hubs tomorrow to discuss trade tariffs. {handle}",
        "Fact-checking the claims made during yesterday's round-table discussion on {topic} and public expenditure. #FactCheck",
        "Voter sentiment survey across 10 battleground districts shows a statistical dead heat between {candidate} and opponents. #Polls",
        "Official notice: Candidate debate on {topic} will be broadcast nationwide with simultaneous sign language interpretation. #Debate2026",
        "The election commission released the final list of polling venues and security protocols for district 4. #ElectionNotice",
        "Summary of candidate remarks regarding agricultural subsidies and regional irrigation projects. #AgriNews",
        "Round 2 of bipartisan town hall discussions scheduled for next Wednesday at 6 PM. Moderated by civic forum.",
        "Reviewing the key legislative bills co-sponsored by {candidate} between 2022 and 2025 regarding {topic}.",
        "Ballot counting procedures and independent audit committee verification guidelines published this morning.",
        "Public hearing on state infrastructure maintenance completed with statements from {candidate}'s liaison.",
        "Televised townhall recap: Comparing speaking times of {candidate} and opponents across 5 policy segments.",
        "District polling data updated as of 12 PM: 42% Candidate A, 40% Candidate B, 18% Candidate C or undecided.",
        "Registration guidelines for overseas citizen voting and mail-in ballot dispatch timelines posted on official portal."
    ]
    
    user_handles = [
        "@voter_voice", "@policy_watch", "@civic_pulse", "@political_insider", "@metro_citizen",
        "@daily_elector", "@campaign_beat", "@democratic_forum", "@grassroots_talk", "@apex_analyst",
        "@liberty_lens", "@voter_chronicle", "@pulse_of_nation", "@state_observer", "@election_daily",
        "@citizen_journal", "@truth_seeker_99", "@econ_observer", "@public_affairs_hub", "@civic_sentinel",
        "@urban_voter", "@swing_state_view", "@insight_politics", "@voter_forum", "@national_digest"
    ]
    
    start_date = datetime(2026, 8, 1)
    
    # 38% Positive, 35% Negative, 27% Neutral
    n_pos = int(num_records * 0.38)
    n_neg = int(num_records * 0.35)
    n_neu = num_records - n_pos - n_neg
    
    sentiment_pool = ["Positive"] * n_pos + ["Negative"] * n_neg + ["Neutral"] * n_neu
    random.shuffle(sentiment_pool)
    
    data = []
    for i, sent in enumerate(sentiment_pool):
        cand = random.choice(candidates)
        topic = random.choice(topics)
        
        if sent == "Positive":
            template = random.choice(positive_phrases)
            likes = random.randint(25, 6200)
            retweets = random.randint(5, 1800)
        elif sent == "Negative":
            template = random.choice(negative_phrases)
            likes = random.randint(30, 6800)
            retweets = random.randint(8, 2100)
        else:
            template = random.choice(neutral_phrases)
            likes = random.randint(10, 950)
            retweets = random.randint(2, 320)
            
        text = template.format(
            candidate=cand["name"],
            topic=topic,
            handle=cand["handle"]
        )
        
        # Add slight conversational variations
        if random.random() < 0.25:
            prefix = random.choice(["Honestly, ", "My opinion: ", "FYI: ", "Just in: ", "Take note: "])
            text = prefix + text
            
        post_date = start_date + timedelta(days=random.randint(0, 60), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        
        data.append({
            "tweet_id": f"TWT-2026-{10001 + i}",
            "created_at": post_date.strftime("%Y-%m-%d %H:%M:%S"),
            "author_handle": random.choice(user_handles),
            "candidate_mentioned": cand["name"],
            "political_party": cand["party"],
            "election_topic": topic,
            "raw_tweet_text": text,
            "likes_count": likes,
            "retweets_count": retweets,
            "ground_truth_sentiment": sent
        })
        
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Generated {len(df)} election tweets at: {output_path}")
    print("Sentiment distribution:\n", df["ground_truth_sentiment"].value_counts())
    return df

if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "election_tweets_raw.csv")
    generate_election_dataset(out_file, num_records=1500)
