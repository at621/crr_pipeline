import pandas as pd
import re
import logging
import tiktoken


def print_summary(df):
    """Print a concise summary of the parsed data"""
    print("\n" + "="*60)
    print("DOCUMENT STRUCTURE SUMMARY")
    print("="*60)
    
    # Document hierarchy
    hierarchy_cols = [col for col in df.columns if col not in 
                     ['Article_Number', 'Article_Heading', 'Text', 'Token_Count', 'Ends_With_Dot', 'Text_With_Pagebreaks']]
    
    if hierarchy_cols:
        print("\nDocument hierarchy:")
        for base in ['Part', 'Title', 'Chapter', 'Section', 'Subsection']:
            if base in df.columns:
                count = df[base].nunique()
                if count > 0:
                    has_heading = f"{base}_Heading" in df.columns
                    print(f"  - {base}: {count} unique {'(with headings)' if has_heading else ''}")
    
    # Validation summary
    print(f"\nValidation summary:")
    print(f"  - Total tokens: {df['Token_Count'].sum():,}")
    print(f"  - Average tokens per article: {df['Token_Count'].mean():.0f}")
    print(f"  - Articles ending with period: {df['Ends_With_Dot'].sum()}/{len(df)}")
    
    # Sample articles
    print(f"\nFirst 5 articles:")
    print(df[['Article_Number', 'Article_Heading', 'Token_Count']].head())
    
    # Numbering patterns
    print("\nNumbering patterns found:")
    for pattern, desc in [(r'\(\d+\)', '(1), (2), (3)'), 
                         (r'\([a-z]\)', '(a), (b), (c)')]:
        count = df['Text'].apply(lambda x: bool(re.search(pattern, x))).sum()
        print(f"  - Articles with {desc}: {count}")


def check_article(df, article_number):
    """Check a specific article for content and numbering"""
    article = df[df['Article_Number'] == str(article_number)]
    if article.empty:
        print(f"Article {article_number} not found")
        return
    
    row = article.iloc[0]
    print(f"\nArticle {article_number}: {row['Article_Heading']}")
    print(f"Tokens: {row['Token_Count']}, Ends with dot: {row['Ends_With_Dot']}")
    
    # Check numbering patterns
    text = row['Text']
    patterns = {
        'Main (1), (2)': re.findall(r'\(\d+\)', text),
        'Letters (a), (b)': re.findall(r'\([a-z]\)', text),
        'Roman (i), (ii)': re.findall(r'\([ivxlcdm]+\)', text, re.IGNORECASE)
    }
    
    print("\nNumbering found:")
    for desc, matches in patterns.items():
        if matches:
            print(f"  {desc}: {', '.join(matches[:5])}{' ...' if len(matches) > 5 else ''}")
    
    print(f"\nText preview:")
    print("-" * 60)
    print(text[:500] + "..." if len(text) > 500 else text)