import {useEffect, useState, useRef} from 'react';
import React from 'react';
import './style.css';
import {formatDate} from '../Tools';
import ErrorTooltip from '../ErrorTooltip';

export default function ForecastProbabilities() {
    const [probs, setProbs] = useState(new Array(14).fill(0));
    const [dates, setDates] = useState(new Array(14).fill(0));

    const [errorText, setErrorText] = useState("");

    useEffect(() => {
        fetch("/api/model/probability")
        .then((response) => response.json())
        .then((data) => {
            setProbs(data.sequence);
            setDates(data.dates);
        })
        .catch((err) => {
            console.error(err);
            setErrorText(err.toString());
        })
    }, []);

    return <>
        <h>Low Gauge Height Forecast</h><ErrorTooltip height={24} text={errorText}></ErrorTooltip>
        <hr></hr>
        <div className="ForecastArea">
            {
                probs.map((element, index) => {
                    return <div key={index} className="ForecastCell">
                        <p>{formatDate(dates[index], false)}</p>
                        <p>{(element * 100).toFixed(0)}%</p>
                        <div>
                            <div style={{height: (element * 100).toFixed(1) + "%"}}/>
                        </div>
                    </div>
                })
            }
        </div>
        </>
}