import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { corsHeaders } from "../_shared/cors.ts";

Deno.serve(async (req) => {
  // This is needed for browser pre-flight requests
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    // 1. Create a Supabase client with the user's authorization
    const supabaseClient = createClient(
      Deno.env.get("SUPABASE_URL"),
      Deno.env.get("SUPABASE_ANON_KEY"),
      {
        global: {
          headers: { Authorization: req.headers.get("Authorization") },
        },
      }
    );

    // 2. Get the user from the token
    const {
      data: { user },
      error: userError,
    } = await supabaseClient.auth.getUser();
    if (userError) throw userError;

    // 3. Get the image file from the request body
    const formData = await req.formData();
    const file = formData.get("file");

    // 4. Upload image to Supabase Storage
    const filePath = `${user.id}/${Date.now()}_${file.name}`;
    const { error: uploadError } = await supabaseClient.storage
      .from("maize-images")
      .upload(filePath, file);
    if (uploadError) throw uploadError;

    // 5. Get the public URL
    const {
      data: { publicUrl },
    } = supabaseClient.storage.from("maize-images").getPublicUrl(filePath);

    // 6. Call your FastAPI backend (from the server)
    const analysisResponse = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData, // Forward the same form data
    });
    if (!analysisResponse.ok) throw new Error("Analysis API request failed.");
    const analysisResult = await analysisResponse.json();

    // 7. Insert the result into the database
    const { error: insertError } = await supabaseClient
      .from("analyses")
      .insert({
        user_id: user.id,
        image_url: publicUrl,
        prediction: analysisResult.prediction,
        confidence: analysisResult.confidence,
        severity_percentage: analysisResult.severity_percentage,
      });
    if (insertError) throw insertError;

    // 8. Return the result to the frontend
    return new Response(JSON.stringify(analysisResult), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 400,
    });
  }
});
