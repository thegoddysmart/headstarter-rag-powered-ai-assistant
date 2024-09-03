import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenerativeAI } from "@google/generative-ai";

// Step 2: Define the system prompt
// const systemPrompt = `
// You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
// For every user question, the top 3 professors that match the user question are returned.
// Use them to answer the question if needed.
// `

const systemPrompt = `
# Rate My Professor Agent System Prompt

You are an AI assistant designed to help students find professors based on their specific queries. Your primary function is to use retrieval-augmented generation (RAG) to provide information about the top 3 most relevant professors for each user question.

## Your Capabilities:
1. Access a comprehensive database of professor information, including:
   - Name and title
   - Department and institution
   - Areas of expertise
   - Course ratings and reviews
   - Research interests and publications
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
   - Department and institution
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
// export async function POST(req) {
//   const data = await req.json()

//   // Step 4: Initialize Pinecone and Google Gemini
//   const pc = new Pinecone({
//     apiKey: process.env.PINECONE_API_KEY,
//   })
//   const index = pc.index('rag').namespace('ns1')

//   // Initialize Google Gemini
//   const genai = new GoogleGenerativeAI(process.env.GOOGLE_GENERATIVE_AI_API_KEY);

//   // Step 5: Process the user’s query
//   const text = data[data.length - 1].content

//   // Create an embedding using Google Gemini
//   const embeddingResult = await genai.embedContent({
//     model: 'models/text-embedding-004',  // Ensure this matches the correct embedding model for Google Gemini
//     content: text,
//     taskType: 'retrieval_document'
//   })

//   const embedding = embeddingResult.data[0].embedding

//   // Step 6: Query Pinecone
//   const results = await index.query({
//     topK: 5,
//     includeMetadata: true,
//     vector: embedding,
//   })

//   // Step 7: Format the results
//   let resultString = ''
//   results.matches.forEach((match) => {
//     resultString += `
//     Returned Results:
//     Professor: ${match.id}
//     Review: ${match.metadata.review}
//     Subject: ${match.metadata.subject}
//     Stars: ${match.metadata.stars}
//     \n\n`
//   })

//   // Step 8: Prepare the Google Gemini request
//   const lastMessage = data[data.length - 1]
//   const lastMessageContent = lastMessage.content + resultString
//   const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

//   // Step 9: Send request to Google Gemini
//   const completion = await genai.generateText({
//     model: 'gemini-1.5-flash',
//     prompt: systemPrompt + '\n' + lastMessageContent,
//     stream: true,
//   })

//   // Step 10: Set up streaming response
//   const stream = new ReadableStream({
//     async start(controller) {
//       const encoder = new TextEncoder()
//       try {
//         for await (const chunk of completion) {
//           const content = chunk.text  // Adjusted to match the expected response structure from Google Gemini
//           if (content) {
//             const text = encoder.encode(content)
//             controller.enqueue(text)
//           }
//         }
//       } catch (err) {
//         controller.error(err)
//       } finally {
//         controller.close()
//       }
//     },
//   })

//   return new NextResponse(stream)
// }


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