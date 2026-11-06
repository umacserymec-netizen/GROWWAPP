# Sources

**Scope decision: the corpus is exactly the 5 Groww help articles listed below.**
No HDFC pages, no mutual-fund scheme pages, and no link following. If an answer
is missing from these pages, the assistant says so rather than fetching another
source.

Official `groww.in` help pages only. No blogs, forums, screenshots, or
third-party aggregators. `sources.csv` is the machine-readable list.

| id | page | official URL | used for |
|---|---|---|---|
| 1 | Do you charge for processing KYC? | https://groww.in/help/my-account/ma-kyc/do-you-charge-for-processing-kyc | KYC processing fees |
| 2 | How to change password on Groww? | https://groww.in/help/my-account/ma-account-details/how-to-change-password-on-groww | password-change steps |
| 3 | What are account opening charges? | https://groww.in/help/stocks/sx-pricing/what-are-account-opening-charges | demat account opening charges |
| 4 | When will the stocks be delivered to my demat account? | https://groww.in/help/stocks/order/when-will-the-stocks-be-delivered-to-my-demat-account-2 | stock delivery timeline |
| 5 | Why is my 'Available to Invest' not the same as my 'Available to Withdraw'? | https://groww.in/help/payments/payments/why-is-my-available-to-invest-not-the-same-as-my-available-to-withdraw-1 | available balance and withdrawal timing |

## Status

All five URLs returned HTTP 200 with extractable article text when the corpus was
selected. `python -m ingest.run fetch` writes the live `http_status`, raw
`sha256`, and `fetched_at` back into `sources.csv`, so the table above is only the
human-readable scope decision.

## Deliberate exclusions

- HDFC and all mutual-fund scheme content.
- Portfolio advice, returns, performance, and recommendations.
- Personal data such as PAN, Aadhaar, account numbers, OTPs, emails, or phone
  numbers.
