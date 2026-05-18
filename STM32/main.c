/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2026 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "i2c.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "OLED.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define RX_BUF_SIZE 128
#define MEDIAN_SAMPLES 5

/* 标定相关 */
#define CALIB_SAMPLE_COUNT 20 // 标定采集帧数
#define H_REAL 198.0f         // 标定用瓶子真实高度 (mm)
#define SKIP_FRAMES_CAL 3     // 标定时舍弃的开始帧数

/* 测量相关 */
#define MEASURE_SAMPLE_CNT 20   // 采集有效帧数
#define MEASURE_TIMEOUT_MS 3000 // 3 秒超时
#define SKIP_FRAMES_MEAS 3      // 测量时舍弃的开始帧数

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* 系统工作状态 */
typedef enum
{
    STATE_IDLE,         // 初始状态，等待第一次按键
    STATE_CAL1_COLLECT, // 第一次标定采集中 (100cm)
    STATE_CAL1_DONE,    // 第一次标定完成，等待第二次按键
    STATE_CAL2_COLLECT, // 第二次标定采集中 (80cm)
    STATE_CAL2_DONE,    // 第二次标定完成，等待第三次按键
    STATE_CAL3_COLLECT, // 第三次标定采集中 (60cm)
    STATE_CAL3_DONE,    // 第三次标定完成，等待第四次按键
    STATE_CAL4_COLLECT, // 第四次标定采集中 (40cm)
    STATE_CAL4_DONE,    // 第四次标定完成，等待第五次按键
    STATE_MEASURE       // 测量模式（支持启停）
} SystemState;
SystemState state = STATE_IDLE;

/* 标定数据 */
float h1_px = 0.0f; // 100cm 处的像素高度
float h2_px = 0.0f; // 80cm  处的像素高度
float h3_px = 0.0f; // 60cm  处的像素高度
float h4_px = 0.0f; // 40cm  处的像素高度

/* 测量结果 */
float D_mm = 0.0f; // 距离 (mm)
float H_mm = 0.0f; // 液面实际高度 (mm)
float L_mm = 0.0f; // 液面到瓶底实际距离 (mm)
float conf = 0.0f; // 置信度
int level_px = 0;  // 液面到瓶底像素距离 (-1 无效)

/* 电流 / 功率 */
float I = 0.0f;    // 电流（由队友提供）
float P = 0.0f;    // 瞬时功耗
float Pmax = 0.0f; // 最大功耗

/* 串口接收相关 */
uint8_t uart_rx_buf[RX_BUF_SIZE];
uint8_t uart_rx_index = 0;
uint8_t uart_frame_ready = 0; // 1 表示收到完整一帧
uint8_t rx_byte = 0;

/* 标定数据缓存 (用于去极值平均) */
float calib_buffer_h[CALIB_SAMPLE_COUNT];
int calib_count = 0;    // 当前标定帧计数
int cal_skip_count = 0; // 标定舍弃帧计数

/* 测量相关变量 */
uint8_t start_flag = 0; // 按键按下后置 1（保留未用）
uint8_t measuring = 0;  // 0 = 停止，1 = 测量中
uint32_t measure_start_time = 0;
uint8_t top_updated = 0;     // 上面两行是否已更新
uint8_t meas_skip_count = 0; // 测量舍弃帧计数

/* 测量数据缓存 */
float D_buf[MEASURE_SAMPLE_CNT];
float H_buf[MEASURE_SAMPLE_CNT];
float L_buf[MEASURE_SAMPLE_CNT];
int measure_cnt = 0;

/* ADC 相关 */
uint16_t adc_raw_value = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* 串口接收中断回调 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1)
    {
        if (rx_byte == '\r')
        {
            HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
            return;
        }
        if (rx_byte == '\n')
        {
            uart_rx_buf[uart_rx_index] = '\0';
            uart_frame_ready = 1;
            uart_rx_index = 0;
        }
        else
        {
            uart_rx_buf[uart_rx_index] = rx_byte;
            uart_rx_index++;
            if (uart_rx_index >= RX_BUF_SIZE - 1)
            {
                uart_rx_buf[uart_rx_index] = '\0';
                uart_frame_ready = 1;
                uart_rx_index = 0;
            }
        }
        HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
    }
}

/* 解析一帧数据，格式: "h,T,B,conf,color,level_dist" */
void parse_data_frame(uint8_t *buffer)
{
    int h_px, T_px, B_px, level_dist_px;
    char color[8];
    float conf_val;

    int ret = sscanf((char *)buffer, "%d,%d,%d,%f,%7[^,],%d",
                     &h_px, &T_px, &B_px, &conf_val, color, &level_dist_px);
    if (ret != 6)
        return;

    conf = conf_val;
    level_px = level_dist_px;

    /* ---------- 标定阶段 ---------- */
    if (state == STATE_CAL1_COLLECT || state == STATE_CAL2_COLLECT ||
        state == STATE_CAL3_COLLECT || state == STATE_CAL4_COLLECT)
    {
        if (cal_skip_count < SKIP_FRAMES_CAL)
        {
            cal_skip_count++;
            return;
        }
        if (h_px > 0 && calib_count < CALIB_SAMPLE_COUNT)
        {
            calib_buffer_h[calib_count++] = (float)h_px;
        }
    }
    /* ---------- 测量阶段 ---------- */
    else if (state == STATE_MEASURE && measuring && !top_updated)
    {
        if (meas_skip_count < SKIP_FRAMES_MEAS)
        {
            meas_skip_count++;
            return;
        }
        if (level_dist_px != -1 && h_px > 0 && measure_cnt < MEASURE_SAMPLE_CNT)
        {
            /*
             * 分段插值：根据 h_px 在 4 段标定值中定位，线性插值得 h_ref 和 D_ref
             * 标定段：h[0]=h1(100cm) < h[1]=h2(80cm) < h[2]=h3(60cm) < h[3]=h4(40cm)
             * 距离：   D_mm[0]=1000   D_mm[1]=800    D_mm[2]=600    D_mm[3]=400
             */
            static const float cal_D[4] = {1000.0f, 800.0f, 600.0f, 400.0f};
            float cal_h[4] = {h1_px, h2_px, h3_px, h4_px};

            float h_ref = h1_px; /* 默认回退 */
            float D_ref = 1000.0f;
            float t;

            if (h_px >= cal_h[3] && cal_h[3] > 0)
            {
                /* 超出最近标定点 → 外推 */
                h_ref = cal_h[3];
                D_ref = cal_D[3];
            }
            else if (h_px <= cal_h[0] && cal_h[0] > 0)
            {
                /* 超出最远标定点 → 外推 */
                h_ref = cal_h[0];
                D_ref = cal_D[0];
            }
            else
            {
                /* 在区间内查找 */
                int i;
                for (i = 0; i < 3; i++)
                {
                    if (h_px >= cal_h[i] && h_px <= cal_h[i + 1] && cal_h[i] > 0 && cal_h[i + 1] > 0)
                    {
                        t = (float)(h_px - cal_h[i]) / (cal_h[i + 1] - cal_h[i]);
                        h_ref = cal_h[i] + (cal_h[i + 1] - cal_h[i]) * t;
                        D_ref = cal_D[i] + (cal_D[i + 1] - cal_D[i]) * t;
                        break;
                    }
                }
            }

            float ratio = H_REAL / h_ref;
            float D_tmp = D_ref * h_ref / h_px;
            float H_tmp = ratio * h_px;
            float L_tmp = ratio * level_dist_px;

            D_buf[measure_cnt] = D_tmp;
            H_buf[measure_cnt] = H_tmp;
            L_buf[measure_cnt] = L_tmp;
            measure_cnt++;
        }
    }
}

// 一阶低通滤波
float low_pass_filter(float new_val)
{
    static float lpf_val = 0.0f;
    lpf_val = lpf_val * 0.95f + new_val * 0.05f;
    return lpf_val;
}

/* 中值滤波读取 ADC（电流） */
uint16_t get_adc_median(void)
{
    uint16_t values[MEDIAN_SAMPLES];
    HAL_ADC_Start(&hadc1);
    for (uint8_t i = 0; i < MEDIAN_SAMPLES; i++)
    {
        if (HAL_ADC_PollForConversion(&hadc1, 10) == HAL_OK)
            values[i] = HAL_ADC_GetValue(&hadc1);
        else
            values[i] = 0;
    }
    HAL_ADC_Stop(&hadc1);

    for (uint8_t i = 0; i < MEDIAN_SAMPLES - 1; i++)
    {
        for (uint8_t j = i + 1; j < MEDIAN_SAMPLES; j++)
        {
            if (values[i] > values[j])
            {
                uint16_t temp = values[i];
                values[i] = values[j];
                values[j] = temp;
            }
        }
    }
    return values[MEDIAN_SAMPLES / 2];
}

/* 读取电流，计算瞬时功率 */
void Read_Current(void)
{
    uint16_t adc_value = get_adc_median();
    adc_raw_value = adc_value;
    float voltage = (adc_value / 4095.0f) * 3.3f;
    voltage = low_pass_filter(voltage); // 加这一行

#define INA240_GAIN 20.0f
#define INA240_VREF 1.63f
#define INA240_RSHUNT 0.1f

    float current = (voltage - INA240_VREF) / (INA240_GAIN * INA240_RSHUNT);
    if (current < -5.0f)
        current = -5.0f;
    if (current > 5.0f)
        current = 5.0f;

    I = current * 1000;
    P = 5.0f * fabsf(current);
}

/* 按键下降沿检测（带消抖） */
uint8_t is_key_pressed(void)
{
    static uint8_t last_state = 1;
    uint8_t current = HAL_GPIO_ReadPin(KEY_START_GPIO_Port, KEY_START_Pin);
    uint8_t pressed = 0;

    if (last_state == 1 && current == 0)
    {
        pressed = 1;
        HAL_Delay(20);
        if (HAL_GPIO_ReadPin(KEY_START_GPIO_Port, KEY_START_Pin) != 0)
            pressed = 0;
    }
    last_state = current;
    return pressed;
}

/* 刷新下面两行：电流和功率 */
void refresh_power_display(void)
{
    float cur_p = 5.0f * fabsf(I) / 1000;
    if (cur_p > Pmax)
        Pmax = cur_p;

    char line3[20];
    sprintf(line3, "I:%.0fmA P:%.2fW", fabsf(I), cur_p);
    OLED_ShowString(3, 1, line3);

    char line4[20];
    sprintf(line4, "Pmax:%.2fW", Pmax);
    OLED_ShowString(4, 1, line4);
}

/* 刷新上面两行：D、H、L */
void update_top_lines(void)
{
    char line1[20];
    sprintf(line1, "D:%.0fmm", D_mm);
    OLED_ShowString(1, 1, line1);

    char line2[20];
    if (H_mm >= 0 && L_mm >= 0)
        sprintf(line2, "H%.0fmm L%.0fmm", H_mm, L_mm);
    else
        sprintf(line2, "No Level");
    OLED_ShowString(2, 1, line2);
}

/* 去极值平均（去掉 remove 个最大和最小） */
float filter_array(float *arr, int n, int remove)
{
    if (n <= 2 * remove)
    {
        /* 数据太少，直接平均 */
        float sum = 0;
        for (int i = 0; i < n; i++)
            sum += arr[i];
        return (n > 0) ? (sum / n) : 0.0f;
    }
    /* 冒泡排序 */
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (arr[i] > arr[j])
            {
                float t = arr[i];
                arr[i] = arr[j];
                arr[j] = t;
            }
        }
    }
    float sum = 0;
    for (int i = remove; i < n - remove; i++)
        sum += arr[i];
    return sum / (n - 2 * remove);
}

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void)
{
    /* USER CODE BEGIN 1 */
    /* USER CODE END 1 */

    /* MCU Configuration--------------------------------------------------------*/
    HAL_Init();
    SystemClock_Config();

    MX_GPIO_Init();
    MX_I2C1_Init();
    MX_USART1_UART_Init();
    MX_ADC1_Init();

    /* USER CODE BEGIN 2 */
    printf("D-TI System Start\r\n");

    if (HAL_ADCEx_Calibration_Start(&hadc1) != HAL_OK)
        Error_Handler();

    OLED_Init();
    OLED_Clear();

    /* 初始界面 */
    OLED_ShowString(1, 1, "Press Key");
    OLED_ShowString(2, 1, "Cal at 100cm");

    HAL_UART_Receive_IT(&huart1, &rx_byte, 1);

    /* 状态变量初始化 */
    state = STATE_IDLE;
    calib_count = 0;
    measuring = 0;
    /* USER CODE END 2 */

    /* Infinite loop */
    /* USER CODE BEGIN WHILE */
    while (1)
    {
        /* -------- 按键处理 -------- */
        if (is_key_pressed())
        {
            switch (state)
            {
            case STATE_IDLE:
                /* 第一次按键 → 开始标定 1 (100cm) */
                state = STATE_CAL1_COLLECT;
                calib_count = 0;
                cal_skip_count = 0;
                OLED_Clear();
                OLED_ShowString(1, 1, "Calibrating...");
                OLED_ShowString(2, 1, "Cal at 100cm");
                break;

            case STATE_CAL1_DONE:
                /* 第二次按键 → 开始标定 2 (80cm) */
                state = STATE_CAL2_COLLECT;
                calib_count = 0;
                cal_skip_count = 0;
                OLED_Clear();
                OLED_ShowString(1, 1, "Calibrating...");
                OLED_ShowString(2, 1, "Cal at 80cm");
                break;

            case STATE_CAL2_DONE:
                /* 第三次按键 → 开始标定 3 (60cm) */
                state = STATE_CAL3_COLLECT;
                calib_count = 0;
                cal_skip_count = 0;
                OLED_Clear();
                OLED_ShowString(1, 1, "Calibrating...");
                OLED_ShowString(2, 1, "Cal at 60cm");
                break;

            case STATE_CAL3_DONE:
                /* 第四次按键 → 开始标定 4 (40cm) */
                state = STATE_CAL4_COLLECT;
                calib_count = 0;
                cal_skip_count = 0;
                OLED_Clear();
                OLED_ShowString(1, 1, "Calibrating...");
                OLED_ShowString(2, 1, "Cal at 40cm");
                break;

            case STATE_CAL4_DONE:
                /* 第五次按键 → 进入测量模式 */
                state = STATE_MEASURE;
                measuring = 1;
                measure_cnt = 0;
                meas_skip_count = 0;
                top_updated = 0;
                Pmax = 0.0f; // 重置最大功率
                measure_start_time = HAL_GetTick();
                OLED_Clear();
                /* 上两行初始为空，下两行随后刷新 */
                update_top_lines();
                break;

            case STATE_MEASURE:
                /* 测量模式下按键 → 切换启停 */
                measuring = !measuring;
                if (measuring)
                {
                    measure_cnt = 0;
                    meas_skip_count = 0;
                    top_updated = 0;
                    Pmax = 0.0f;
                    measure_start_time = HAL_GetTick();
                    OLED_Clear();
                    update_top_lines();
                }
                else
                {
                    OLED_Clear();
                    OLED_ShowString(1, 1, "PAUSED");
                }
                break;

            default:
                /* 采集中按键无效 */
                break;
            }
        }

        /* -------- 处理串口数据帧 -------- */
        if (uart_frame_ready)
        {
            uart_frame_ready = 0;
            parse_data_frame(uart_rx_buf);
            uart_rx_index = 0;
        }

        /* -------- 检查标定是否完成 -------- */
        if (state == STATE_CAL1_COLLECT && calib_count >= CALIB_SAMPLE_COUNT)
        {
            h1_px = filter_array(calib_buffer_h, CALIB_SAMPLE_COUNT, 3);
            state = STATE_CAL1_DONE;
            OLED_Clear();
            OLED_ShowString(1, 1, "Cal at 100cm OK");
            OLED_ShowString(2, 1, "Press Key -> 80cm");
        }
        else if (state == STATE_CAL2_COLLECT && calib_count >= CALIB_SAMPLE_COUNT)
        {
            h2_px = filter_array(calib_buffer_h, CALIB_SAMPLE_COUNT, 3);
            state = STATE_CAL2_DONE;
            OLED_Clear();
            OLED_ShowString(1, 1, "Cal at 80cm OK");
            OLED_ShowString(2, 1, "Press Key -> 60cm");
        }
        else if (state == STATE_CAL3_COLLECT && calib_count >= CALIB_SAMPLE_COUNT)
        {
            h3_px = filter_array(calib_buffer_h, CALIB_SAMPLE_COUNT, 3);
            state = STATE_CAL3_DONE;
            OLED_Clear();
            OLED_ShowString(1, 1, "Cal at 60cm OK");
            OLED_ShowString(2, 1, "Press Key -> 40cm");
        }
        else if (state == STATE_CAL4_COLLECT && calib_count >= CALIB_SAMPLE_COUNT)
        {
            h4_px = filter_array(calib_buffer_h, CALIB_SAMPLE_COUNT, 3);
            state = STATE_CAL4_DONE;
            OLED_Clear();
            OLED_ShowString(1, 1, "Cal at 40cm OK");
            OLED_ShowString(2, 1, "Press to Start");
        }

        /* -------- 测量状态下的处理 -------- */
        if (state == STATE_MEASURE && measuring)
        {
            /* 定期刷新下面两行（200ms） */
            static uint32_t last_power_refresh = 0;
            uint32_t now = HAL_GetTick();
            if (now - last_power_refresh >= 200)
            {
                last_power_refresh = now;
                Read_Current();
                refresh_power_display();
            }

            /* 上面两行数据收集与超时处理 */
            if (!top_updated)
            {
                if (measure_cnt >= MEASURE_SAMPLE_CNT)
                {
                    /* 收满 20 帧 */
                    D_mm = filter_array(D_buf, MEASURE_SAMPLE_CNT, 2);
                    H_mm = filter_array(H_buf, MEASURE_SAMPLE_CNT, 2);
                    L_mm = filter_array(L_buf, MEASURE_SAMPLE_CNT, 2);
                    update_top_lines();
                    top_updated = 1;
                }
                else if (HAL_GetTick() - measure_start_time > MEASURE_TIMEOUT_MS)
                {
                    /* 超时 3 秒 */
                    if (measure_cnt > 0)
                    {
                        D_mm = filter_array(D_buf, measure_cnt, 2);
                        H_mm = filter_array(H_buf, measure_cnt, 2);
                        L_mm = filter_array(L_buf, measure_cnt, 2);
                    }
                    else
                    {
                        D_mm = 0.0f;
                        H_mm = 0.0f;
                        L_mm = 0.0f;
                    }
                    update_top_lines();
                    top_updated = 1;
                }
            }
        }
        else
        {
            /* 非测量状态时简单延时，降低 CPU 占用 */
            HAL_Delay(50);
        }
        HAL_Delay(1);
        /* USER CODE END WHILE */
        /* USER CODE BEGIN 3 */
    }
    /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
    RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_ON;
    RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
        Error_Handler();

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
        Error_Handler();

    PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_ADC;
    PeriphClkInit.AdcClockSelection = RCC_ADCPCLK2_DIV6;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
        Error_Handler();
}

/* USER CODE BEGIN 4 */
/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void)
{
    __disable_irq();
    while (1)
    {
    }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
    /* USER CODE BEGIN 6 */
    /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
