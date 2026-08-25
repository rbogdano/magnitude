import { describe, expect, it } from "vitest"
import { localMaxOutputTokens } from "./local-provider-offerings"

describe("localMaxOutputTokens", () => {
  it("never advertises the whole window", () => {
    // The agent passes this straight through as `max_tokens`, and prompt plus completion share the
    // served context. Advertising the window is a request the engine must refuse on turn one.
    for (const contextLength of [4_096, 8_192, 16_384, 32_768, 131_072]) {
      expect(localMaxOutputTokens(contextLength)).toBeLessThan(contextLength)
    }
  })

  it("leaves the growing part of the window the larger share", () => {
    // The conversation and the tool schemas are what expand; one turn's answer is not.
    expect(localMaxOutputTokens(16_384)).toBe(4_096)
    expect(localMaxOutputTokens(32_768)).toBe(8_192)
    expect(localMaxOutputTokens(131_072)).toBe(8_192)
  })

  it("keeps a small context usable", () => {
    // Halving a tiny window would advertise a completion too short to be worth generating.
    expect(localMaxOutputTokens(1_024)).toBe(1_024)
    expect(localMaxOutputTokens(512)).toBe(1_024)
  })
})
