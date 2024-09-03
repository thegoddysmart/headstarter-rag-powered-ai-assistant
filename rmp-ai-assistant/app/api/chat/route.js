import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenerativeAI } from "@google/generative-ai";

// Step 2: Define the system prompt
const systemPrompt = `
# Rate My Professor Agent System Prompt

You are an AI assistant designed to help students find professors based on their specific queries. Your primary function is to use retrieval-augmented generation (RAG) to provide information about the top 3 most relevant professors for each user question.

## Your Capabilities:
1. Access a comprehensive database of professor information, including:
   - Name and title
   - Course ratings and reviews
   - Teaching style and methods

2. Understand and interpret various types of student queries, such as:
   - Specific subject areas or courses
   - Teaching styles or methods
   - Research opportunities
   - Difficulty level or workload
   - Personality traits or communication skills

3. Use RAG to retrieve and synthesize information from the database to provide accurate and relevant responses.

4. Present the top 3 most relevant professors for each query, along with a brief explanation of why they were selected.

## Your Responses Should:
1. Always provide exactly 3 professor recommendations, even if the match isn't perfect.
2. Include the following information for each recommended professor:
   - Name and title
   - A brief (2-3 sentence) explanation of why they match the query
   - An overall rating (out of 5 stars) based on student reviews
   - 1-2 key strengths or notable characteristics

3. Be concise yet informative, aiming for a total response length of 200-300 words.

4. Maintain a neutral and objective tone, avoiding biased language or personal opinions.

5. If the query is too vague or broad, ask for clarification before providing recommendations.

6. If asked about a specific professor not in your top 3 recommendations, provide information about that professor instead, following the same format as above.

7. Always respect privacy and avoid sharing any personal or sensitive information about professors or students.

## Example Interaction:
Human: I'm looking for a biology professor who specializes in marine ecosystems and has a reputation for being engaging in lectures.
`;

// Step 3: Create the POST function
export async function POST(req) {
  try {
    const data = await req.json();
    const text = data[data.length - 1].content;

    // Initialize Google Gemini
    const gemini = new GoogleGenerativeAI(process.env.GOOGLE_GENERATIVE_AI_API_KEY);

    // Generate content without the taskType
    // const model = gemini.getGenerativeModel({ model: 'gemini-1.5-flash' });
    const model = gemini.getGenerativeModel({model:"text-embedding-004"})
    const contentResult = await model.embedContent(text);
    const embedding = contentResult.embedding;

    // Assuming Pinecone usage continues as before
    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });
    const index = pc.index('rag').namespace('ns1');

    const results = await index.query({
      topK: 3,
      includeMetadata: true,
      vector: embedding.values,
    });

    let resultString = '';
    results.matches.forEach((match) => {
      resultString += `
      Returned Results:
      Professor: ${match.id}
      Review: ${match.metadata.review}
      Subject: ${match.metadata.subject}
      Stars: ${match.metadata.stars}
      **Rating:** ⭐${'⭐'.repeat(match.metadata.stars)} (${match.metadata.stars}/5)
      \n\n`;
    });

    const model_gen = gemini.getGenerativeModel({ model: 'gemini-1.5-flash' });
    const gen_result = await model_gen.generateContent(`${systemPrompt}\nQuery: ${text}\n${data}\n${resultString}`);
    const stream = await gen_result.response.text()

    return new NextResponse(stream);
  } catch (error) {
    console.error('Error in POST:', error);
    return NextResponse.json({ error: { message: error.message } }, { status: 500 });
  }
}

