# Bid Datasets Directory

This folder contains the benchmark bid packages provided for the Emplay AI Engineering assignment. Each subfolder represents one independent bid and is ingested and processed as a single unit.

---

## 📁 Bid1: Dallas ISD — Student and Staff Computing Devices
**Solicitation ID**: `JA-207652` | **Portal Sourcing ID**: `#168884`

| File Name | Format | Document Type | Description |
|---|---|---|---|
| `JA-207652 Student and Staff Computing Devices FINAL.pdf` | PDF | `RFP` | Base RFP specification, proposal requirements, general terms & conditions |
| `Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf` | PDF | `ADDENDUM` | Addendum 1: Clarifications, Q&A responses, OEM warranty rule, 5-day repair SLA |
| `Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf` | PDF | `ADDENDUM` | Addendum 2: Formal submission deadline extension to **July 9, 2024 at 2:00 PM CST** |
| `Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html` | HTML | `BID_PAGE` | Online portal solicitation listing and pre-bidding event schedule |

---

## 📁 Bid2: State of Maryland — Dell Laptops w/ Extended Warranty
**Solicitation ID**: `PORFP #E20P4600040` | **eMMA Project**: `BPM044557`

| File Name | Format | Document Type | Description |
|---|---|---|---|
| `PORFP_-_Dell_Laptop_Final.pdf` | PDF | `RFP` | Purchase Order Request for Proposals under Hardware Master Contract (FA I & FA V) |
| `Dell_Laptop_Specs.pdf` | PDF | `SPECS` | Detailed hardware configuration & SKU list for Dell Latitude 5550 (SI# CC7802) |
| `Contract_Affidavit.pdf` | PDF | `AFFIDAVIT` | Mandatory State of Maryland legal contract affidavit |
| `Mercury_Affidavit.pdf` | PDF | `AFFIDAVIT` | Mandatory State of Maryland environmental mercury affidavit |
| `Dell Laptops w_Extended Warranty - Bid Information - {3} _ BidNet Direct.html` | HTML | `BID_PAGE` | Online portal posting from BidNet Direct |

---

## 🚀 Adding New Bids (Incremental Indexing)
To add any new bid folder, simply create a new directory:
```
data/bids/Bid3/
  ├── Your_RFP.pdf
  ├── Addendum_1.pdf
  └── Portal_Listing.html
```
Then run:
```bash
python main.py index data/bids/Bid3 --bid-id Bid3
```
Incremental indexing ensures existing bids (`Bid1`, `Bid2`) are not re-indexed.
