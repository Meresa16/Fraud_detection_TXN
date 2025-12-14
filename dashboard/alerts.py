def fraud_alert(txn, model):
    pred = model.predict(txn)[0]
    if pred == 1:
        return "⚠️ FRAUD ALERT: Transaction Blocked"
    return "✅ Transaction Approved"
